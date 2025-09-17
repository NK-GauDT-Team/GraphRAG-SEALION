import os
import json
import asyncio
import threading
import logging
from typing import Optional, List, Any, Tuple, Dict

import gradio as gr
import websockets
import requests
from openai import OpenAI
from PyPDF2 import PdfReader

from langchain.schema import Document
from langchain.llms.base import LLM

from graph_rag import GraphRAG

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

graph_rag_system: Optional[GraphRAG] = None
documents_processed: bool = False
websocket_server = None
connected_clients = set()

# Search-agent config
SEARXNG_URL = os.getenv("SEARXNG_URL", "https://searxng.hweeren.com/")
OVERPASS_URLS = [
    os.getenv("OVERPASS_URL") or "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.openstreetmap.fr/api/interpreter",
    "https://z.overpass-api.de/api/interpreter",
]
USER_AGENT = "GraphRAG-Pharmacy/1.0 (+inventory)"

# Google Directions — REQUIRED for distance/ETA
GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY")
GOOGLE_DIRECTIONS_URL = "https://maps.googleapis.com/maps/api/directions/json"

# ──────────────────────────────────────────────────────────────────────────────
# SEALION LangChain Wrapper
# ──────────────────────────────────────────────────────────────────────────────
class SEALIONWrapper(LLM):
    client: Any = None
    model_name: str = "aisingapore/Llama-SEA-LION-v3.5-70B-R"

    def __init__(self, api_key: str, **kwargs):
        super().__init__(**kwargs)
        self.client = OpenAI(api_key=api_key, base_url="https://api.sea-lion.ai/v1")

    @property
    def _llm_type(self) -> str:
        return "sealion"

    def _call(self, prompt: str, stop=None, run_manager=None, **kwargs: Any) -> str:
        system_msg = (
            "You are a medical AI assistant with access to a GraphRAG medical database.\n"
            "Return ONLY valid JSON that matches EXACTLY the schema below. Do NOT add any prose outside the JSON.\n\n"
            "Language policy:\n"
            "- Let L_user be the language of the user's latest message.\n"
            "- All natural-language fields MUST be written in L_user.\n"
            "- If the user mixes languages, choose the language that dominates most tokens.\n"
            "- Do not translate medicine brand names or international dose units; translate descriptions/labels only.\n"
            "- Never fall back to English unless the user wrote in English.\n\n"
            "JSON schema:\n"
            "{\n"
            '  "analysis": string,\n'
            '  "severity": "low" | "medium" | "high",\n'
            '  "medicines": [\n'
            '    {"name": string, "dosage": string, "description": string, "localAvailability": string},\n'
            '    {"name": string, "dosage": string, "description": string, "localAvailability": string}\n'
            "  ],\n"
            '  "disclaimer": string,\n'
            '  "seekEmergencyCare": boolean,\n'
            '  "language": string,\n'
            '  "uiLabels": {\n'
            '     "recommended": string,\n'
            '     "dosage": string,\n'
            '     "availability": string,\n'
            '     "emergency": string\n'
            "  }\n"
            "}\n\n"
            "Content rules:\n"
            "- Recommend 2–4 medicines ONLY if present in the GraphRAG context; do not invent medicines.\n"
            "- Use medical knowledge for analysis/safety; keep medicine names exactly as in context.\n"
            "- Keep uiLabels concise and natural for a UI in L_user.\n"
        )

        def chat_once(user_msg: str) -> str:
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "system", "content": system_msg},
                          {"role": "user", "content": user_msg}],
                extra_body={"chat_template_kwargs": {"thinking_mode": "off"},
                            "cache": {"no-cache": True}},
                temperature=0.1,
                max_tokens=1500,
            )
            return completion.choices[0].message.content or ""

        try:
            out = chat_once(prompt).strip()
        except Exception as e:
            logger.error(f"SEALION API error: {e}")
            return (
                '{"analysis":"Error calling SEALION.","severity":"low","medicines":[],"disclaimer":"",'
                '"seekEmergencyCare":false,"language":"en","uiLabels":{"recommended":"Recommended medicines:",'
                '"dosage":"Dosage","availability":"Check availability","emergency":"Seek emergency medical care immediately"}}'
            )

        if not (out.startswith("{") and out.endswith("}")):
            try:
                out = chat_once(
                    "Convert the following answer into EXACTLY the required JSON schema and return ONLY the JSON:\n\n"
                    f"{out}"
                ).strip()
            except Exception as e:
                logger.error(f"SEALION repair error: {e}")
                out = (
                    '{"analysis":"Unable to produce JSON.","severity":"low","medicines":[],"disclaimer":"",'
                    '"seekEmergencyCare":false,"language":"en","uiLabels":{"recommended":"Recommended medicines:",'
                    '"dosage":"Dosage","availability":"Check availability","emergency":"Seek emergency medical care immediately"}}'
                )
        return out


# ──────────────────────────────────────────────────────────────────────────────
# Models & PDFs
# ──────────────────────────────────────────────────────────────────────────────
def initialize_models() -> Tuple[Optional[GraphRAG], str]:
    try:
        from sentence_transformers import SentenceTransformer

        class DirectEmbeddings:
            def __init__(self):
                self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
            def embed_documents(self, texts: List[str]):
                return self.model.encode(texts, convert_to_tensor=False).tolist()
            def embed_query(self, text: str):
                return self.model.encode(text, convert_to_tensor=False).tolist()

        embedding_model = DirectEmbeddings()
        logger.info("✅ Embedding model loaded successfully")

        api_key = os.getenv("SEALION_API_KEY")
        if not api_key:
            logger.warning("⚠️ SEALION_API_KEY not set — calls will fail.")
        try:
            llm = SEALIONWrapper(api_key=api_key)
            logger.info("✅ SEALION LLM initialized")
            _ = llm._call("Hi")
        except Exception as e:
            msg = f"❌ Error initializing SEALION: {e}"
            logger.error(msg)
            return None, msg

        try:
            graph_rag = GraphRAG(embedding_model, llm)
            logger.info("✅ GraphRAG system initialized")
        except Exception as e:
            msg = f"❌ Error initializing GraphRAG: {e}"
            logger.error(msg)
            return None, msg

        return graph_rag, "✅ Models initialized successfully!"
    except ImportError as e:
        msg = f"❌ Missing dependency: {e}. Run: pip install sentence-transformers"
        logger.error(msg)
        return None, msg
    except Exception as e:
        msg = f"❌ Unexpected error: {e}"
        logger.error(msg)
        return None, msg

def process_pdf_files(files) -> str:
    global graph_rag_system, documents_processed
    if not files:
        return "❌ No files uploaded. Please upload PDF files first."
    if graph_rag_system is None:
        graph_rag_system, init_msg = initialize_models()
        if graph_rag_system is None:
            return init_msg
    try:
        documents: List[Document] = []
        processed_files: List[str] = []
        failed_files: List[str] = []

        for file_path in files:
            try:
                reader = PdfReader(file_path)
                text = ""
                for page in reader.pages:
                    try:
                        text += page.extract_text() or ""
                    except Exception as e:
                        logger.warning(f"PDF page extract error: {e}")
                if text.strip():
                    documents.append(Document(page_content=text, metadata={"source": os.path.basename(file_path)}))
                    processed_files.append(os.path.basename(file_path))
                else:
                    failed_files.append(os.path.basename(file_path))
            except Exception as e:
                failed_files.append(os.path.basename(file_path))
                logger.error(f"Error processing {file_path}: {e}")

        if not documents:
            return "❌ No valid PDF content found in uploaded files."

        graph_rag_system.process_documents(documents)
        documents_processed = True

        try:
            num_nodes = len(graph_rag_system.knowledge_graph.graph.nodes())
            num_edges = len(graph_rag_system.knowledge_graph.graph.edges())
        except Exception:
            num_nodes = num_edges = 0

        msg = (
            f"✅ Successfully processed {len(documents)} documents!\n\n"
            f"📊 Graph: {num_nodes} nodes / {num_edges} edges\n"
            f"Files processed: {', '.join(processed_files)}"
        )
        if failed_files:
            msg += f"\nFailed files: {', '.join(failed_files)}"
        msg += "\n\n🌐 WebSocket server is ready to receive queries from clients!"
        return msg
    except Exception as e:
        msg = f"❌ Unexpected error processing documents: {e}"
        logger.error(msg)
        return msg

# ──────────────────────────────────────────────────────────────────────────────
# Helpers: Overpass, SearXNG, Google Directions
# ──────────────────────────────────────────────────────────────────────────────
def searxng_search(query: str, language: Optional[str] = None, pageno: int = 1, timeout: int = 10) -> Dict[str, Any]:
    url = SEARXNG_URL.rstrip("/") + "/search"
    params = {"q": query, "format": "json", "pageno": pageno, "safesearch": 1}
    if language:
        params["language"] = language
    r = requests.get(url, params=params, headers={"User-Agent": USER_AGENT}, timeout=timeout)
    r.raise_for_status()
    return r.json()

def overpass_pharmacies(center, radius_km, limit=50):
    R = max(100, min(50000, int(radius_km * 1000)))
    q = f"""
    [out:json][timeout:60];
    (
      node["amenity"="pharmacy"](around:{R},{center['lat']},{center['lon']});
      way["amenity"="pharmacy"](around:{R},{center['lat']},{center['lon']});
    );
    out tags center {min(limit, 200)};
    """
    last = None
    for url in OVERPASS_URLS:
        try:
            r = requests.post(url, data=q, headers={"User-Agent": USER_AGENT}, timeout=60)
            r.raise_for_status()
            return r.json().get("elements", [])
        except Exception as e:
            last = e
            logger.warning(f"Overpass fail {url}: {e}")
            continue
    raise last or RuntimeError("Overpass failed")

def overpass_convenience(center, radius_km, limit=50):
    R = max(100, min(50000, int(radius_km * 1000)))
    q = f"""
    [out:json][timeout:60];
    (
      node["shop"="convenience"](around:{R},{center['lat']},{center['lon']});
      way["shop"="convenience"](around:{R},{center['lat']},{center['lon']});
    );
    out tags center {min(limit, 200)};
    """
    last = None
    for url in OVERPASS_URLS:
        try:
            r = requests.post(url, data=q, headers={"User-Agent": USER_AGENT}, timeout=60)
            r.raise_for_status()
            return r.json().get("elements", [])
        except Exception as e:
            last = e
            logger.warning(f"Overpass fail {url}: {e}")
            continue
    raise last or RuntimeError("Overpass failed")

def google_route_km_min(
    lat1: float, lon1: float, lat2: float, lon2: float,
    mode: str = "walking",
    timeout: int = 15,
) -> Optional[Tuple[float, float]]:
    """
    Returns (distance_km, duration_min) using Google Directions API.
    mode: walking | driving | transit | bicycling
    """
    if not GOOGLE_MAPS_API_KEY:
        logger.debug("Google API key missing; cannot compute route.")
        return None
    try:
        params = {
            "origin": f"{lat1},{lon1}",
            "destination": f"{lat2},{lon2}",
            "mode": mode,
            "units": "metric",
            "alternatives": "false",
            "key": GOOGLE_MAPS_API_KEY,
        }
        r = requests.get(GOOGLE_DIRECTIONS_URL, params=params, timeout=timeout)
        r.raise_for_status()
        js = r.json()
        if js.get("status") != "OK" or not js.get("routes"):
            logger.debug(f"Google Directions non-OK: {js.get('status')}")
            return None
        leg = js["routes"][0]["legs"][0]
        dist_km = round((leg["distance"]["value"] or 0) / 1000.0, 2)  # meters → km
        dur_min = round((leg["duration"]["value"] or 0) / 60.0, 1)    # seconds → min
        return dist_km, dur_min
    except Exception as e:
        logger.debug(f"Google Directions error: {e}")
        return None

# ──────────────────────────────────────────────────────────────────────────────
# GPS-only pharmacy search
# ──────────────────────────────────────────────────────────────────────────────
ALLOWED_TRAVEL_MODES = {"walking", "driving"}  # easy to extend later

async def run_pharmacy_search(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    GPS-only.
    payload: {
      "user_location": {"lat": float, "lon": float, "accuracy_m"?: number, "ts"?: number},  # REQUIRED
      "radius_km": int,
      "limit": int,
      "language": str | None,
      "medicines": [{"name": str, "dosage": str?}, ...],
      "mode": "walking" | "driving"   # NEW: optional, default "walking"
    }
    """
    user_location = payload.get("user_location")
    radius_km = int(payload.get("radius_km") or 8)
    limit = int(payload.get("limit") or 12)
    language = payload.get("language")
    medicines = payload.get("medicines") or []
    mode_in = (payload.get("mode") or "walking").lower().strip()
    mode = mode_in if mode_in in ALLOWED_TRAVEL_MODES else "walking"

    if not medicines:
        return {"center": None, "city": None, "pharmacies": [], "message": "No medicines provided."}

    if not user_location or "lat" not in user_location or "lon" not in user_location:
        return {
            "center": None,
            "city": None,
            "pharmacies": [],
            "source": "gps-only",
            "fallbackUsed": False,
            "radius_km": radius_km,
            "message": "GPS not provided. Tap 'Use my location'.",
        }

    # Pull accuracy if provided
    try:
        accuracy_m = float(user_location.get("accuracy_m") or 0.0)
    except Exception:
        accuracy_m = 0.0

    center = {"lat": float(user_location["lat"]), "lon": float(user_location["lon"])}
    logger.info(f"🔎 GPS center used: {center} (accuracy_m={accuracy_m}) mode={mode}")

    # Reject obviously-bad fixes to avoid routing from a wrong district
    if accuracy_m and accuracy_m > 1000:
        return {
            "center": center,
            "city": None,
            "pharmacies": [],
            "source": "gps-only",
            "fallbackUsed": False,
            "radius_km": radius_km,
            "message": f"Location too imprecise (~{int(accuracy_m)}m). Try again near a window with Precise Location on.",
        }

    # 1) Nearby pharmacies via Overpass around GPS
    try:
        elems = await asyncio.to_thread(overpass_pharmacies, center, radius_km, max(100, limit * 4))
    except Exception as e:
        logger.error(f"Overpass error: {e}")
        elems = []

    candidates: List[Dict[str, Any]] = []
    for e in elems:
        tags = e.get("tags", {}) or {}
        lat = (e.get("lat") or e.get("center", {}).get("lat"))
        lon = (e.get("lon") or e.get("center", {}).get("lon"))
        if lat is None or lon is None:
            continue

        addr_parts = [
            tags.get("addr:housenumber"),
            tags.get("addr:street"),
            tags.get("addr:suburb"),
            tags.get("addr:city"),
            tags.get("addr:state"),
            tags.get("addr:postcode"),
            tags.get("addr:country"),
        ]
        address = ", ".join([p for p in addr_parts if p])

        candidates.append({
            "id": f"osm_{e.get('type','n')}_{e.get('id')}",
            "name": tags.get("name") or "Pharmacy",
            "address": address or tags.get("addr:full"),
            "latitude": str(lat),
            "longitude": str(lon),
            "country": tags.get("addr:country"),
            "city": tags.get("addr:city"),
            "phoneNumber": tags.get("phone") or tags.get("contact:phone"),
            "openingHours": tags.get("opening_hours"),
        })

    if not candidates:
        return {
            "center": center,
            "city": None,
            "pharmacies": [],
            "source": "overpass (none)",
            "fallbackUsed": False,
            "radius_km": radius_km,
            "message": "No nearby pharmacies found from Overpass.",
        }

    # 2) Google route distance/ETA from GPS (mode selected)
    async def add_route_metrics(p: Dict[str, Any]) -> Dict[str, Any]:
        try:
            lat2, lon2 = float(p["latitude"]), float(p["longitude"])
            res = await asyncio.to_thread(
                google_route_km_min,
                center["lat"], center["lon"], lat2, lon2,
                mode, 15
            )
            if res is not None:
                dist_km, dur_min = res
                p["distance_km"] = dist_km
                p["duration_min"] = dur_min
            else:
                p["distance_km"] = None
            return p
        except Exception as e:
            logger.debug(f"Route metrics error for {p.get('id')}: {e}")
            p["distance_km"] = None
            return p

    candidates = await asyncio.gather(*[add_route_metrics(p) for p in candidates])

    # 3) Availability via SearXNG (best-effort heuristic)
    def score_availability(text: str) -> Optional[str]:
        t = (text or "").lower()
        if any(k in t for k in ["in stock", "available", "còn hàng", "có sẵn", "thêm vào giỏ", "add to cart"]):
            return "in_stock"
        if any(k in t for k in ["out of stock", "hết hàng", "ngừng kinh doanh"]):
            return "out_of_stock"
        return None

    def status_to_score(status: str) -> float:
        return {
            "in_stock": 0.92,
            "likely_in_stock": 0.70,
            "call_to_confirm": 0.50,
            "out_of_stock": 0.0,
        }.get(status, 0.0)

    async def enrich_pharmacy(p: Dict[str, Any]) -> Dict[str, Any]:
        matches = []
        for m in medicines:
            med_name = m.get("name") or ""
            base_queries: List[str] = [
                f'{med_name} "{p["name"]}"',
                f"{med_name}"
            ]

            best = None
            for q in base_queries[:3]:
                try:
                    js = await asyncio.to_thread(searxng_search, q, language)
                except Exception:
                    continue
                for item in js.get("results", []):
                    text = f'{item.get("title","")} {item.get("content","")}'
                    status = score_availability(text) or "likely_in_stock"
                    cand = {
                        "medicineName": med_name,
                        "status": status,
                        "score": status_to_score(status),
                        "url": item.get("url"),
                    }
                    if (best is None) or (cand["score"] > best["score"]):
                        best = cand
            if best is None:
                best = {"medicineName": med_name, "status": "call_to_confirm", "score": status_to_score("call_to_confirm")}
            matches.append(best)

        best_score = max((mm["score"] for mm in matches), default=0.0)
        p_out = {**p, "matches": matches, "bestScore": best_score}
        return p_out

    enriched = await asyncio.gather(*[enrich_pharmacy(p) for p in candidates])

    def good_count(ph: Dict[str, Any]) -> int:
        return sum(1 for mm in ph.get("matches", []) if mm["status"] != "out_of_stock")

    enriched.sort(
        key=lambda ph: (
            -good_count(ph),
            -(ph.get("bestScore") or 0.0),
            ph.get("distance_km") if ph.get("distance_km") is not None else 1e9,
        )
    )

    return {
        "center": center,
        "city": None,
        "pharmacies": enriched[:limit],
        "source": f"overpass+searxng+google-directions ({mode})",
        "fallbackUsed": False,
        "radius_km": radius_km,
    }


async def run_convenience_search(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    GPS-only.
    payload: {
      "user_location": {"lat": float, "lon": float, "accuracy_m"?: number, "ts"?: number},  # REQUIRED
      "radius_km": int,
      "limit": int,
      "language": str | None,
      "medicines": [{"name": str, "dosage": str?}, ...],
      "mode": "walking" | "driving"   # NEW: optional, default "walking"
    }
    """
    user_location = payload.get("user_location")
    radius_km = int(payload.get("radius_km") or 8)
    limit = int(payload.get("limit") or 12)
    language = payload.get("language")
    medicines = payload.get("medicines") or []
    mode_in = (payload.get("mode") or "walking").lower().strip()
    mode = mode_in if mode_in in ALLOWED_TRAVEL_MODES else "walking"

    if not medicines:
        return {"center": None, "city": None, "convenience": [], "message": "No local remedy provided."}

    if not user_location or "lat" not in user_location or "lon" not in user_location:
        return {
            "center": None,
            "city": None,
            "convenience": [],
            "source": "gps-only",
            "fallbackUsed": False,
            "radius_km": radius_km,
            "message": "GPS not provided. Tap 'Use my location'.",
        }

    # Pull accuracy if provided
    try:
        accuracy_m = float(user_location.get("accuracy_m") or 0.0)
    except Exception:
        accuracy_m = 0.0

    center = {"lat": float(user_location["lat"]), "lon": float(user_location["lon"])}
    logger.info(f"🔎 GPS center used: {center} (accuracy_m={accuracy_m}) mode={mode}")

    # Reject obviously-bad fixes to avoid routing from a wrong district
    if accuracy_m and accuracy_m > 1000:
        return {
            "center": center,
            "city": None,
            "pharmacies": [],
            "source": "gps-only",
            "fallbackUsed": False,
            "radius_km": radius_km,
            "message": f"Location too imprecise (~{int(accuracy_m)}m). Try again near a window with Precise Location on.",
        }

    # 1) Nearby pharmacies via Overpass around GPS
    try:
        elems = await asyncio.to_thread(overpass_convenience, center, radius_km, max(100, limit * 4))
    except Exception as e:
        logger.error(f"Overpass error: {e}")
        elems = []

    candidates: List[Dict[str, Any]] = []
    for e in elems:
        tags = e.get("tags", {}) or {}
        lat = (e.get("lat") or e.get("center", {}).get("lat"))
        lon = (e.get("lon") or e.get("center", {}).get("lon"))
        if lat is None or lon is None:
            continue

        addr_parts = [
            tags.get("addr:housenumber"),
            tags.get("addr:street"),
            tags.get("addr:suburb"),
            tags.get("addr:city"),
            tags.get("addr:state"),
            tags.get("addr:postcode"),
            tags.get("addr:country"),
        ]
        address = ", ".join([p for p in addr_parts if p])

        candidates.append({
            "id": f"osm_{e.get('type','n')}_{e.get('id')}",
            "name": tags.get("name") or "convenience",
            "address": address or tags.get("addr:full"),
            "latitude": str(lat),
            "longitude": str(lon),
            "country": tags.get("addr:country"),
            "city": tags.get("addr:city"),
            "phoneNumber": tags.get("phone") or tags.get("contact:phone"),
            "openingHours": tags.get("opening_hours"),
        })

    if not candidates:
        return {
            "center": center,
            "city": None,
            "convenience": [],
            "source": "overpass (none)",
            "fallbackUsed": False,
            "radius_km": radius_km,
            "message": "No nearby convenience store found from Overpass.",
        }

    # 2) Google route distance/ETA from GPS (mode selected)
    async def add_route_metrics(p: Dict[str, Any]) -> Dict[str, Any]:
        try:
            lat2, lon2 = float(p["latitude"]), float(p["longitude"])
            res = await asyncio.to_thread(
                google_route_km_min,
                center["lat"], center["lon"], lat2, lon2,
                mode, 15
            )
            if res is not None:
                dist_km, dur_min = res
                p["distance_km"] = dist_km
                p["duration_min"] = dur_min
            else:
                p["distance_km"] = None
            return p
        except Exception as e:
            logger.debug(f"Route metrics error for {p.get('id')}: {e}")
            p["distance_km"] = None
            return p

    candidates = await asyncio.gather(*[add_route_metrics(p) for p in candidates])

    # 3) Availability via SearXNG (best-effort heuristic)
    def score_availability(text: str) -> Optional[str]:
        t = (text or "").lower()
        if any(k in t for k in ["in stock", "available", "add to cart"]):
            return "in_stock"
        if any(k in t for k in ["out of stock"]):
            return "out_of_stock"
        return None

    def status_to_score(status: str) -> float:
        return {
            "in_stock": 0.92,
            "likely_in_stock": 0.70,
            "call_to_confirm": 0.50,
            "out_of_stock": 0.0,
        }.get(status, 0.0)

    async def enrich_convenience(p: Dict[str, Any]) -> Dict[str, Any]:
        matches = []
        for m in medicines:
            med_name = m.get("name") or ""
            base_queries: List[str] = [
                f'{med_name} "{p["name"]}"',
                f"{med_name}"
            ]

            best = None
            for q in base_queries[:3]:
                try:
                    js = await asyncio.to_thread(searxng_search, q, language)
                except Exception:
                    continue
                for item in js.get("results", []):
                    text = f'{item.get("title","")} {item.get("content","")}'
                    status = score_availability(text) or "likely_in_stock"
                    cand = {
                        "medicineName": med_name,
                        "status": status,
                        "score": status_to_score(status),
                        "url": item.get("url"),
                    }
                    if (best is None) or (cand["score"] > best["score"]):
                        best = cand
            if best is None:
                best = {"medicineName": med_name, "status": "call_to_confirm", "score": status_to_score("call_to_confirm")}
            matches.append(best)

        best_score = max((mm["score"] for mm in matches), default=0.0)
        p_out = {**p, "matches": matches, "bestScore": best_score}
        return p_out

    enriched = await asyncio.gather(*[enrich_convenience(p) for p in candidates])

    def good_count(ph: Dict[str, Any]) -> int:
        return sum(1 for mm in ph.get("matches", []) if mm["status"] != "out_of_stock")

    enriched.sort(
        key=lambda ph: (
            -good_count(ph),
            -(ph.get("bestScore") or 0.0),
            ph.get("distance_km") if ph.get("distance_km") is not None else 1e9,
        )
    )

    return {
        "center": center,
        "city": None,
        "convenience": enriched[:limit],
        "source": f"overpass+searxng+google-directions ({mode})",
        "fallbackUsed": False,
        "radius_km": radius_km,
    }


# ──────────────────────────────────────────────────────────────────────────────
# WebSocket + Admin UI
# ──────────────────────────────────────────────────────────────────────────────
async def process_client_message(message: str, websocket) -> str:
    global graph_rag_system, documents_processed
    try:
        data = json.loads(message)
        msg_type = (data.get("type") or "").lower()

        if msg_type == "pharmacy_search":
            rid = data.get("rid")
            payload = data.get("payload") or {}
            logger.info(f"🔎 Pharmacy search rid={rid} payload={str(payload)[:200]}...")
            result = await run_pharmacy_search(payload)
            return json.dumps({"type": "pharmacy_search_result", "rid": rid, **result})
        
        if msg_type == "convenience_search":
            rid = data.get("rid")
            payload = data.get("payload") or {}
            logger.info(f"Convenience store search rid={rid} payload={str(payload)[:200]}...")
            result = await run_convenience_search(payload)
            return json.dumps({"type": "convenience_search_result", "rid": rid, **result})

        if not documents_processed or graph_rag_system is None:
            return json.dumps({"type": "error", "message": "System not ready. Please upload and process documents."})

        if msg_type in ("", "query"):
            query = data.get("message", "")
            if not query.strip():
                return json.dumps({"type": "error", "message": "Please enter a valid question."})
            answer, traversal_path, context_data = graph_rag_system.query(query)
            try:
                s = (answer or "").strip()
                if s.startswith("{") and s.endswith("}"):
                    medical_response = json.loads(s)
                    if all(k in medical_response for k in ["analysis", "medicines", "severity"]):
                        response = {
                            "type": "medical_response",
                            "message": medical_response.get("analysis", ""),
                            "medicines": medical_response.get("medicines", []),
                            "severity": medical_response.get("severity", "low"),
                            "disclaimer": medical_response.get("disclaimer", ""),
                            "seekEmergencyCare": medical_response.get("seekEmergencyCare", False),
                            "uiLabels": medical_response.get("uiLabels", {}),
                            "language": medical_response.get("language"),
                            "metadata": {"traversal_nodes": len(traversal_path), "sources_used": context_data.get("num_sources", 0)},
                        }
                    else:
                        response = {"type": "response", "message": answer,
                                    "metadata": {"traversal_nodes": len(traversal_path), "sources_used": context_data.get("num_sources", 0)}}
                else:
                    response = {"type": "response", "message": answer,
                                "metadata": {"traversal_nodes": len(traversal_path), "sources_used": context_data.get("num_sources", 0)}}
            except json.JSONDecodeError:
                response = {"type": "response", "message": answer,
                            "metadata": {"traversal_nodes": len(traversal_path), "sources_used": context_data.get("num_sources", 0)}}
            return json.dumps(response)

        return json.dumps({"type": "error", "message": f"Unknown message type: {msg_type}"})
    except json.JSONDecodeError:
        return json.dumps({"type": "error", "message": "Invalid message format"})
    except Exception as e:
        logger.error(f"Error processing WebSocket message: {e}")
        return json.dumps({"type": "error", "message": f"Error processing query: {e}"})

async def websocket_handler(websocket, path: Optional[str] = None):
    connected_clients.add(websocket)
    try:
        await websocket.send(json.dumps({"type": "connection", "message": "Connected", "status": "ready" if documents_processed else "waiting_for_documents"}))
        async for message in websocket:
            response = await process_client_message(message, websocket)
            await websocket.send(response)
    finally:
        if websocket in connected_clients:
            connected_clients.remove(websocket)

def start_websocket_server():
    async def serve():
        global websocket_server
        websocket_server = await websockets.serve(websocket_handler, "0.0.0.0", 8765, ping_interval=20, ping_timeout=10)
        await asyncio.Future()
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(serve())

def reset_system() -> str:
    global graph_rag_system, documents_processed
    graph_rag_system = None
    documents_processed = False
    return "System reset. Please upload documents again."

def get_system_status() -> str:
    global graph_rag_system, documents_processed, connected_clients
    parts: List[str] = []
    if graph_rag_system is None:
        parts.append("Models not initialized")
    elif not documents_processed:
        parts.append("Models ready, no documents processed")
    else:
        try:
            num_nodes = len(graph_rag_system.knowledge_graph.graph.nodes())
            num_edges = len(graph_rag_system.knowledge_graph.graph.edges())
            parts.append(f"System ready - {num_nodes} nodes, {num_edges} edges")
        except Exception as e:
            parts.append(f"System ready but error getting stats: {e}")
    parts.append(f"WebSocket: {len(connected_clients)} clients connected")
    return "\n".join(parts)

def create_admin_interface():
    with gr.Blocks(title="GraphRAG Admin Panel", theme=gr.themes.Soft()) as demo:
        gr.HTML("<h1>🔧 GraphRAG Admin Panel</h1><p>WebSocket Server: ws://localhost:8765</p>")
        with gr.Row():
            with gr.Column():
                file_upload = gr.File(label="Upload PDF Files", file_types=[".pdf"], file_count="multiple")
                process_btn = gr.Button("Process Documents", variant="primary")
                upload_status = gr.Textbox(label="Processing Status", interactive=False, lines=10)
            with gr.Column():
                system_status = gr.Textbox(label="Current Status", interactive=False, lines=6, value=get_system_status())
                refresh_status_btn = gr.Button("Refresh Status")
                reset_btn = gr.Button("Reset System", variant="stop")
        process_btn.click(fn=process_pdf_files, inputs=[file_upload], outputs=[upload_status], show_progress=True)
        reset_btn.click(fn=reset_system, outputs=[upload_status])
        refresh_status_btn.click(fn=get_system_status, outputs=[system_status])
        demo.load(fn=get_system_status, outputs=[system_status])
    return demo

if __name__ == "__main__":
    if not os.getenv("SEALION_API_KEY"):
        print("Warning: SEALION_API_KEY not set!\nSet it with: export SEALION_API_KEY=your-api-key")
    if not GOOGLE_MAPS_API_KEY:
        print("Warning: GOOGLE_MAPS_API_KEY not set! Distance/ETA will be missing.")
    threading.Thread(target=start_websocket_server, daemon=True).start()
    create_admin_interface().launch(server_name="0.0.0.0", server_port=7860, share=False, show_error=True)
