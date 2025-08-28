import os
import json
import asyncio
import threading
import logging
from typing import Optional, List, Any, Tuple

import gradio as gr
import websockets
from openai import OpenAI
from PyPDF2 import PdfReader

from langchain.schema import Document
from langchain.llms.base import LLM

# Your GraphRAG system (assumed available in your env)
from graph_rag import GraphRAG

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Globals
# -----------------------------------------------------------------------------
graph_rag_system: Optional[GraphRAG] = None
documents_processed: bool = False
websocket_server = None
connected_clients = set()


# -----------------------------------------------------------------------------
# SEALION LangChain Wrapper
# -----------------------------------------------------------------------------
class SEALIONWrapper(LLM):
    """Custom LangChain wrapper for SEALION API."""
    client: Any = None
    model_name: str = "aisingapore/Llama-SEA-LION-v3.5-8B-R"

    def __init__(self, api_key: str, **kwargs):
        super().__init__(**kwargs)
        self.client = OpenAI(api_key=api_key, base_url="https://api.sea-lion.ai/v1")

    @property
    def _llm_type(self) -> str:
        return "sealion"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> str:
        """
        Returns a JSON string. If the first try isn't valid JSON, we repair once.
        JSON schema includes uiLabels for front-end rendering (no hardcoding).
        """
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
            '  "analysis": string,                       // in L_user\n'
            '  "severity": "low" | "medium" | "high",\n'
            '  "medicines": [                           // names untransformed; desc/dosage text in L_user\n'
            '    {"name": string, "dosage": string, "description": string, "localAvailability": string},\n'
            '    {"name": string, "dosage": string, "description": string, "localAvailability": string}\n'
            "  ],\n"
            '  "disclaimer": string,                    // in L_user\n'
            '  "seekEmergencyCare": boolean,\n'
            '  "language": string,                      // ISO 639-1 code you detected for L_user\n'
            '  "uiLabels": {                            // short UI labels in L_user (no examples)\n'
            '     "recommended": string,                // heading for the list of medicines\n'
            '     "dosage": string,                     // label preceding dosage text\n'
            '     "availability": string,               // label/phrase for availability badge\n'
            '     "emergency": string                   // short emergency warning\n'
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
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg},
                ],
                extra_body={
                    "chat_template_kwargs": {"thinking_mode": "off"},
                    "cache": {"no-cache": True},
                },
                temperature=0.1,
                max_tokens=1500,
            )
            return completion.choices[0].message.content or ""

        try:
            out = chat_once(prompt).strip()
        except Exception as e:
            logger.error(f"SEALION API error on first try: {e}")
            return f'{{"analysis":"Error calling SEALION.","severity":"low","medicines":[],"disclaimer":"","seekEmergencyCare":false,"language":"en","uiLabels":{"recommended":"Recommended medicines:","dosage":"Dosage","availability":"Check availability","emergency":"Seek emergency medical care immediately"}}}'

        # If not valid JSON, ask the model to convert to JSON once
        if not (out.startswith("{") and out.endswith("}")):
            repair_prompt = (
                "Convert the following answer into EXACTLY the required JSON schema, "
                "keeping all text in the user's language. Return ONLY the JSON:\n\n"
                f"{out}"
            )
            try:
                out = chat_once(repair_prompt).strip()
            except Exception as e:
                logger.error(f"SEALION API error during repair: {e}")
                # Safe minimal JSON
                out = (
                    '{"analysis":"Unable to produce JSON.","severity":"low","medicines":[],'
                    '"disclaimer":"","seekEmergencyCare":false,"language":"en",'
                    '"uiLabels":{"recommended":"Recommended medicines:","dosage":"Dosage",'
                    '"availability":"Check availability","emergency":"Seek emergency medical care immediately"}}'
                )
        return out


# -----------------------------------------------------------------------------
# Model Initialization
# -----------------------------------------------------------------------------
def initialize_models() -> Tuple[Optional[GraphRAG], str]:
    """Initialize embeddings + LLM + GraphRAG."""
    try:
        # Use SentenceTransformers directly (simple + avoids LangChain embedding quirks)
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

        # LLM
        try:
            llm = SEALIONWrapper(api_key=api_key)
            logger.info("✅ SEALION LLM initialized")
            # Smoke test
            test_response = llm._call("Hi")
            if not (test_response.strip().startswith("{") and test_response.strip().endswith("}")):
                logger.warning("⚠️ SEALION test did not return JSON. Continuing anyway.")
        except Exception as e:
            msg = f"❌ Error initializing SEALION: {e}"
            logger.error(msg)
            return None, msg

        # GraphRAG
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


# -----------------------------------------------------------------------------
# Document Processing
# -----------------------------------------------------------------------------
def process_pdf_files(files) -> str:
    """Process uploaded PDF files and build knowledge graph."""
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
                logger.info(f"Processing file: {file_path}")

                reader = PdfReader(file_path)
                text = ""
                for page_num, page in enumerate(reader.pages):
                    try:
                        text += page.extract_text() or ""
                    except Exception as e:
                        logger.warning(f"Error extracting page {page_num} from {file_path}: {e}")
                        continue

                if text.strip():
                    documents.append(
                        Document(page_content=text, metadata={"source": os.path.basename(file_path)})
                    )
                    processed_files.append(os.path.basename(file_path))
                    logger.info(f"✅ Extracted text from {os.path.basename(file_path)}")
                else:
                    failed_files.append(os.path.basename(file_path))
                    logger.warning(f"❌ No extractable text in {os.path.basename(file_path)}")

            except Exception as e:
                failed_files.append(os.path.basename(file_path))
                logger.error(f"Error processing {file_path}: {e}")
                continue

        if not documents:
            return "❌ No valid PDF content found in uploaded files."

        # Build knowledge graph
        try:
            logger.info("Building knowledge graph...")
            graph_rag_system.process_documents(documents)
            documents_processed = True
            logger.info("✅ Knowledge graph built successfully")
        except Exception as e:
            msg = f"❌ Error building knowledge graph: {e}"
            logger.error(msg)
            return msg

        # Graph stats
        try:
            num_nodes = len(graph_rag_system.knowledge_graph.graph.nodes())
            num_edges = len(graph_rag_system.knowledge_graph.graph.edges())
        except Exception as e:
            logger.warning(f"Error getting graph stats: {e}")
            num_nodes = num_edges = 0

        success_msg = (
            f"✅ Successfully processed {len(documents)} documents!\n\n"
            f"📊 **Graph Statistics:**\n"
            f"- **Nodes:** {num_nodes}\n"
            f"- **Edges:** {num_edges}\n"
            f"- **Files processed:** {', '.join(processed_files)}"
        )
        if failed_files:
            success_msg += f"\n- **Failed files:** {', '.join(failed_files)}"
        success_msg += "\n\n🌐 WebSocket server is ready to receive queries from clients!"

        return success_msg

    except Exception as e:
        msg = f"❌ Unexpected error processing documents: {e}"
        logger.error(msg)
        return msg


# -----------------------------------------------------------------------------
# Query Handling (WebSocket)
# -----------------------------------------------------------------------------
async def process_client_message(message: str, websocket) -> str:
    """Process message from client and return JSON str to send back."""
    global graph_rag_system, documents_processed

    try:
        data = json.loads(message)
        query = data.get("message", "")

        if not documents_processed or graph_rag_system is None:
            return json.dumps(
                {
                    "type": "error",
                    "message": "System not ready. Please wait for admin to upload and process documents.",
                }
            )

        if not query.strip():
            return json.dumps({"type": "error", "message": "Please enter a valid question."})

        logger.info(f"Processing client query: {query}")

        # GraphRAG query should return (answer, traversal_path, context_data)
        answer, traversal_path, context_data = graph_rag_system.query(query)

        # Try to parse medical JSON
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
                        "metadata": {
                            "traversal_nodes": len(traversal_path),
                            "sources_used": context_data.get("num_sources", 0),
                        },
                    }
                else:
                    # JSON but not medical schema -> treat as plain text
                    response = {
                        "type": "response",
                        "message": answer,
                        "metadata": {
                            "traversal_nodes": len(traversal_path),
                            "sources_used": context_data.get("num_sources", 0),
                        },
                    }
            else:
                # Plain text
                response = {
                    "type": "response",
                    "message": answer,
                    "metadata": {
                        "traversal_nodes": len(traversal_path),
                        "sources_used": context_data.get("num_sources", 0),
                    },
                }
        except json.JSONDecodeError:
            response = {
                "type": "response",
                "message": answer,
                "metadata": {
                    "traversal_nodes": len(traversal_path),
                    "sources_used": context_data.get("num_sources", 0),
                },
            }

        logger.info("✅ Query processed successfully via WebSocket")
        return json.dumps(response)

    except json.JSONDecodeError:
        return json.dumps({"type": "error", "message": "Invalid message format"})
    except Exception as e:
        logger.error(f"Error processing WebSocket message: {e}")
        return json.dumps({"type": "error", "message": f"Error processing query: {e}"})


# -----------------------------------------------------------------------------
# WebSocket Server
# -----------------------------------------------------------------------------
async def websocket_handler(websocket, path: Optional[str] = None):
    """Handle WebSocket connections (compatible with websockets v10–v12)."""
    connected_clients.add(websocket)
    logger.info(f"Client connected. Total clients: {len(connected_clients)}")

    try:
        await websocket.send(
            json.dumps(
                {
                    "type": "connection",
                    "message": "Connected to GraphRAG server",
                    "status": "ready" if documents_processed else "waiting_for_documents",
                }
            )
        )

        async for message in websocket:
            response = await process_client_message(message, websocket)
            await websocket.send(response)

    except websockets.exceptions.ConnectionClosed:
        logger.info("Client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        if websocket in connected_clients:
            connected_clients.remove(websocket)
        logger.info(f"Client removed. Total clients: {len(connected_clients)}")


def start_websocket_server():
    """Start the WebSocket server in a separate thread."""
    async def serve():
        global websocket_server
        websocket_server = await websockets.serve(
            websocket_handler,
            "0.0.0.0",
            8765,
            ping_interval=20,
            ping_timeout=10,
        )
        logger.info("🌐 WebSocket server started on ws://0.0.0.0:8765")
        await asyncio.Future()  # run forever

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(serve())


# -----------------------------------------------------------------------------
# Admin Utilities + Gradio UI
# -----------------------------------------------------------------------------
def reset_system() -> str:
    """Reset the entire system."""
    global graph_rag_system, documents_processed
    graph_rag_system = None
    documents_processed = False
    logger.info("🔄 System reset")
    return "🔄 System reset. Please upload documents again."


def get_system_status() -> str:
    """Get current system status."""
    global graph_rag_system, documents_processed, connected_clients

    status_parts: List[str] = []

    if graph_rag_system is None:
        status_parts.append("❌ Models not initialized")
    elif not documents_processed:
        status_parts.append("⚠️ Models ready, no documents processed")
    else:
        try:
            num_nodes = len(graph_rag_system.knowledge_graph.graph.nodes())
            num_edges = len(graph_rag_system.knowledge_graph.graph.edges())
            status_parts.append(f"✅ System ready - {num_nodes} nodes, {num_edges} edges")
        except Exception as e:
            status_parts.append(f"⚠️ System ready but error getting stats: {e}")

    status_parts.append(f"🌐 WebSocket: {len(connected_clients)} clients connected")
    return "\n".join(status_parts)


def create_admin_interface():
    """Create the Gradio admin interface for document management."""
    with gr.Blocks(
        title="GraphRAG Admin Panel",
        theme=gr.themes.Soft(),
        css="""
        .gradio-container { max-width: 1200px !important; }
        .header { text-align: center; margin-bottom: 30px; }
        .status-box { background-color: #f0f2f6; padding: 15px; border-radius: 10px; margin: 10px 0; }
        """,
    ) as demo:

        gr.HTML("""
        <div class="header">
            <h1>🔧 GraphRAG Admin Panel</h1>
            <p>Document Management and System Administration</p>
            <p style="color: #666;">WebSocket Server: ws://localhost:8765</p>
        </div>
        """)

        with gr.Row():
            with gr.Column(scale=1):
                gr.HTML("<h2>📄 Document Management</h2>")

                file_upload = gr.File(
                    label="Upload PDF Files",
                    file_types=[".pdf"],
                    file_count="multiple",
                )

                process_btn = gr.Button("🚀 Process Documents", variant="primary", size="lg")

                upload_status = gr.Textbox(
                    label="Processing Status",
                    interactive=False,
                    lines=10,
                    value="Upload PDF files and click 'Process Documents' to build the knowledge graph...",
                )

            with gr.Column(scale=1):
                gr.HTML("<h2>📊 System Status</h2>")

                system_status = gr.Textbox(
                    label="Current Status",
                    interactive=False,
                    lines=4,
                    value="❌ Models not initialized\n🌐 WebSocket: 0 clients connected",
                )

                with gr.Row():
                    refresh_status_btn = gr.Button("🔄 Refresh Status")
                    reset_btn = gr.Button("🗑️ Reset System", variant="stop")

                gr.HTML("""
                <div class="status-box">
                    <h3>ℹ️ Admin Instructions:</h3>
                    <ul>
                        <li>Upload PDF documents and process them to build the knowledge graph</li>
                        <li>The WebSocket server runs on port 8765</li>
                        <li>Clients can connect to ws://[server-ip]:8765 to send queries</li>
                        <li>Monitor connected clients and system status here</li>
                        <li>The knowledge graph persists until system reset</li>
                    </ul>
                </div>
                """)

                gr.HTML("""
                <div class="status-box">
                    <h3>🔡 WebSocket Message Format:</h3>
                    <pre style="background: #f5f5f5; padding: 10px; border-radius: 5px;">
// Client sends:
{
  "message": "User's question here"
}

// Server responds (medical result):
{
  "type": "medical_response",
  "message": "analysis text ...",
  "medicines": [{...}],
  "severity": "low|medium|high",
  "disclaimer": "...",
  "seekEmergencyCare": false,
  "uiLabels": { "recommended": "...", "dosage": "...", "availability": "...", "emergency": "..." },
  "language": "id|vi|ms|en|..."
}
                    </pre>
                </div>
                """)

        # Events
        process_btn.click(fn=process_pdf_files, inputs=[file_upload], outputs=[upload_status], show_progress=True)
        reset_btn.click(fn=reset_system, outputs=[upload_status])
        refresh_status_btn.click(fn=get_system_status, outputs=[system_status])

        # Auto-refresh on load
        demo.load(fn=get_system_status, outputs=[system_status])

    return demo


# -----------------------------------------------------------------------------
# Entrypoint
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    if not os.getenv("SEALION_API_KEY"):
        print("⚠️  Warning: SEALION_API_KEY not set!")
        print("Set it with: export SEALION_API_KEY=your-api-key")

    # Start WebSocket server in a background thread
    websocket_thread = threading.Thread(target=start_websocket_server, daemon=True)
    websocket_thread.start()

    # Launch Gradio admin UI
    demo = create_admin_interface()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False, show_error=True)
