import logging
from typing import List, Tuple, Optional, Any
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
import uuid

logger = logging.getLogger(__name__)

class LangChainEmbeddingWrapper:
    """Wrapper to make our embedding model compatible with LangChain/Chroma."""
    
    def __init__(self, embedding_model):
        self.embedding_model = embedding_model
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of texts."""
        return self.embedding_model.embed_documents(texts)
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query."""
        return self.embedding_model.embed_query(text)

class DocumentProcessor:
    """Process documents for the GraphRAG system."""
    
    def __init__(self, embedding_model):
        """
        Initialize DocumentProcessor.
        
        Args:
            embedding_model: Embedding model for creating vectors
        """
        self.embedding_model = embedding_model
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        
    def process_documents(self, documents: List[Document]) -> Tuple[List[Document], Any]:
        """
        Process documents into chunks and create vector store.
        
        Args:
            documents: List of documents to process
            
        Returns:
            Tuple of (document_splits, vector_store)
        """
        try:
            # Split documents into chunks
            splits = self.text_splitter.split_documents(documents)
            logger.info(f"Split {len(documents)} documents into {len(splits)} chunks")
            
            # Create vector store directly with the embedding model
            # The embedding model already has embed_documents and embed_query methods
            vector_store = Chroma.from_documents(
                documents=splits,
                embedding=self.embedding_model,  # Use the model directly
                collection_name=f"doc_collection_{uuid.uuid4().hex[:8]}"
            )
            
            logger.info("Created vector store successfully")
            
            return splits, vector_store
            
        except Exception as e:
            logger.error(f"Error processing documents: {e}")
            raise