import logging
from typing import Dict, List, Optional, Tuple, Any
from document_processor import DocumentProcessor
from knowledge_graph import KnowledgeGraph
from query_engine import QueryEngine

logger = logging.getLogger(__name__)

class GraphRAG:
    """Main class for the Graph-based RAG system using NetworkX."""
    
    def __init__(self, embeddings, llm):
        """
        Initialize GraphRAG with required models and processors.
        
        Args:
            embeddings: Embedding model
            llm: Language model
        """
        self.llm = llm
        self.embedding_model = embeddings
        self.document_processor = DocumentProcessor(self.embedding_model)
        self.knowledge_graph = KnowledgeGraph(self.embedding_model, self.llm)
        self.query_engine = None

    def process_documents(self, documents: List[Any]):
        """
        Process documents and initialize QueryEngine.
        
        Args:
            documents: List of documents to process
        """
        try:
            # Process documents to create chunks and vector store
            splits, vector_store = self.document_processor.process_documents(documents)
            
            # Build knowledge graph
            self.knowledge_graph.build_graph(splits)
            
            # Initialize query engine
            self.query_engine = QueryEngine(
                vector_store=vector_store,
                knowledge_graph=self.knowledge_graph,
                llm=self.llm,
                embedding_model=self.embedding_model
            )
            
            logger.info("Documents processed and QueryEngine initialized")
        
        except Exception as e:
            logger.error(f"Error in process_documents: {e}")
            raise

    def query(self, query: str) -> Tuple[str, List[int], Dict[str, Any]]:
        """
        Execute a query.
        
        Args:
            query: Query string
            
        Returns:
            Tuple containing:
                - final answer string
                - list of traversal path node IDs
                - dictionary with context and token usage information
        """
        if not self.query_engine:
            raise ValueError("Documents must be processed before querying")

        try:
            return self.query_engine.query(query)
        except Exception as e:
            logger.error(f"Error in GraphRAG query: {e}")
            return f"Error processing query: {str(e)}", [], {}