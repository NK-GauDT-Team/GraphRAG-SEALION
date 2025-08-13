import logging
from typing import Dict, Any, List, Tuple
import numpy as np
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

logger = logging.getLogger(__name__)

class QueryEngine:
    """Engine for processing queries using vector store and knowledge graph."""
    
    def __init__(self, vector_store, knowledge_graph, llm, embedding_model):
        """
        Initialize the QueryEngine.
        
        Args:
            vector_store: Vector store instance
            knowledge_graph: Knowledge graph instance
            llm: Language model instance
            embedding_model: Embedding model instance
        """
        self.vector_store = vector_store
        self.knowledge_graph = knowledge_graph
        self.llm = llm
        self.embedding_model = embedding_model
        
        # Create prompt template
        self.prompt_template = PromptTemplate(
            template="""Based on the following context from documents, please answer the question comprehensively.

Context:
{context}

Question: {question}

Answer: Please provide a detailed answer based on the context above. If the context doesn't contain enough information to fully answer the question, say so and provide what information is available.""",
            input_variables=["context", "question"]
        )
        
        self.llm_chain = LLMChain(llm=self.llm, prompt=self.prompt_template)

    def query(self, query: str) -> Tuple[str, List[int], Dict[str, Any]]:
        """
        Execute a query using both vector search and graph traversal.
        
        Args:
            query: Query string
            
        Returns:
            Tuple of (answer, traversal_path, context_data)
        """
        try:
            # Step 1: Vector similarity search
            vector_results = self.vector_store.similarity_search(query, k=3)
            
            # Step 2: Graph-based expansion
            query_embedding = self.embedding_model.embed_documents([query])[0]
            relevant_nodes = self.knowledge_graph.get_relevant_nodes(
                np.array(query_embedding), top_k=3
            )
            
            # Step 3: Collect context from both approaches
            context_pieces = []
            traversal_path = []
            
            # Add vector search results
            for doc in vector_results:
                context_pieces.append(doc.page_content)
            
            # Add graph traversal results
            for node_id, similarity in relevant_nodes:
                node_content = self.knowledge_graph.get_node_content(node_id)
                if node_content and node_content not in context_pieces:
                    context_pieces.append(node_content)
                    traversal_path.append(node_id)
                
                # Get neighbors for richer context
                neighbors = self.knowledge_graph.get_neighbors(node_id, max_neighbors=2)
                for neighbor_id in neighbors:
                    neighbor_content = self.knowledge_graph.get_node_content(neighbor_id)
                    if neighbor_content and neighbor_content not in context_pieces:
                        context_pieces.append(neighbor_content)
                        traversal_path.append(neighbor_id)
            
            # Step 4: Generate answer using LLM
            context = "\n\n".join(context_pieces[:5])  # Limit context length
            
            answer = self.llm_chain.run(
                context=context,
                question=query
            )
            
            context_data = {
                "num_sources": len(context_pieces),
                "vector_results": len(vector_results),
                "graph_nodes": len(relevant_nodes)
            }
            
            return answer.strip(), traversal_path, context_data
            
        except Exception as e:
            logger.error(f"Error in query execution: {e}")
            return f"Error processing query: {str(e)}", [], {}