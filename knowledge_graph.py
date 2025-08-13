import networkx as nx
import logging
from typing import List, Dict, Any, Tuple
from langchain.schema import Document
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

class KnowledgeGraph:
    """NetworkX-based Knowledge Graph implementation for RAG."""
    
    def __init__(self, embeddings: Any, llm: Any):
        """
        Initialize the KnowledgeGraph.
        
        Args:
            embeddings: Embedding model
            llm: Language model
        """
        self.graph = nx.Graph()
        self.embeddings = embeddings
        self.llm = llm
        self.node_counter = 0
        
    def build_graph(self, splits: List[Document]) -> None:
        """
        Build the knowledge graph from document splits.
        
        Args:
            splits: List of document splits
        """
        if not splits:
            logger.warning("No documents provided for graph building")
            return

        try:
            # Add nodes for each document split
            self._add_nodes(splits)
            
            # Create embeddings for similarity calculation
            embeddings = self._create_embeddings(splits)
            
            # Add edges based on similarity
            self._add_edges(embeddings)
            
            logger.info(f"Built knowledge graph with {len(self.graph.nodes())} nodes and {len(self.graph.edges())} edges")
            
        except Exception as e:
            logger.error(f"Error building graph: {e}")
            raise

    def _add_nodes(self, splits: List[Document]) -> None:
        """Add nodes to the graph from document splits."""
        for i, split in enumerate(splits):
            if not split.page_content.strip():
                continue
                
            node_id = self.node_counter
            self.graph.add_node(
                node_id,
                content=split.page_content,
                metadata=split.metadata if hasattr(split, 'metadata') else {},
                index=i
            )
            self.node_counter += 1
            
        logger.info(f"Added {len(splits)} nodes to the graph")

    def _create_embeddings(self, splits: List[Document]) -> np.ndarray:
        """Create embeddings for all document splits."""
        try:
            texts = [split.page_content for split in splits if split.page_content.strip()]
            embeddings = self.embeddings.embed_documents(texts)
            return np.array(embeddings)
        except Exception as e:
            logger.error(f"Error creating embeddings: {e}")
            raise

    def _add_edges(self, embeddings: np.ndarray, threshold: float = 0.3) -> None:
        """Add edges between similar nodes."""
        try:
            # Calculate cosine similarity matrix
            similarity_matrix = cosine_similarity(embeddings)
            
            # Add edges for similar documents
            nodes = list(self.graph.nodes())
            for i in range(len(nodes)):
                for j in range(i + 1, len(nodes)):
                    similarity = similarity_matrix[i][j]
                    
                    if similarity > threshold:
                        self.graph.add_edge(
                            nodes[i],
                            nodes[j],
                            weight=similarity,
                            similarity=similarity
                        )
            
            logger.info(f"Added {len(self.graph.edges())} edges to the graph")
            
        except Exception as e:
            logger.error(f"Error adding edges: {e}")
            raise

    def get_neighbors(self, node_id: int, max_neighbors: int = 5) -> List[int]:
        """Get the most similar neighbors of a node."""
        try:
            if node_id not in self.graph:
                return []
            
            # Get neighbors sorted by similarity
            neighbors = []
            for neighbor in self.graph.neighbors(node_id):
                edge_data = self.graph.get_edge_data(node_id, neighbor)
                similarity = edge_data.get('similarity', 0)
                neighbors.append((neighbor, similarity))
            
            # Sort by similarity and return top neighbors
            neighbors.sort(key=lambda x: x[1], reverse=True)
            return [neighbor[0] for neighbor in neighbors[:max_neighbors]]
            
        except Exception as e:
            logger.error(f"Error getting neighbors: {e}")
            return []

    def get_node_content(self, node_id: int) -> str:
        """Get the content of a node."""
        try:
            return self.graph.nodes[node_id].get('content', '')
        except KeyError:
            logger.error(f"Node {node_id} not found")
            return ""

    def get_relevant_nodes(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Tuple[int, float]]:
        """Find the most relevant nodes for a query."""
        try:
            node_similarities = []
            
            for node_id in self.graph.nodes():
                node_content = self.get_node_content(node_id)
                if node_content:
                    # Get embedding for node content
                    node_embedding = self.embeddings.embed_documents([node_content])[0]
                    
                    # Calculate similarity
                    similarity = cosine_similarity([query_embedding], [node_embedding])[0][0]
                    node_similarities.append((node_id, similarity))
            
            # Sort by similarity and return top_k
            node_similarities.sort(key=lambda x: x[1], reverse=True)
            return node_similarities[:top_k]
            
        except Exception as e:
            logger.error(f"Error finding relevant nodes: {e}")
            return []