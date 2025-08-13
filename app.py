import gradio as gr
import tempfile
import os
from graph_rag import GraphRAG
from langchain_community.embeddings import HuggingFaceEmbeddings  # FIXED IMPORT
from langchain.llms.base import LLM
from openai import OpenAI
from PyPDF2 import PdfReader
from langchain.schema import Document
import logging
from typing import Optional, List, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables
graph_rag_system = None
documents_processed = False

class SEALIONWrapper(LLM):
    """Custom LangChain wrapper for SEALION API."""
    
    client: Any = None  # Add this field declaration
    model_name: str = "aisingapore/Llama-SEA-LION-v3.5-8B-R"  # Add this field declaration
    
    def __init__(self, api_key: str, **kwargs):  # Add **kwargs
        super().__init__(**kwargs)  # Pass kwargs to parent
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.sea-lion.ai/v1"
        )
    
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
        try:
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                extra_body={
                    "chat_template_kwargs": {
                        "thinking_mode": "off"
                    },
                    "cache": {
                        "no-cache": True
                    }
                },
                temperature=0.1,
                max_tokens=1000
            )
            return completion.choices[0].message.content
        except Exception as e:
            logger.error(f"SEALION API error: {e}")
            return f"Error calling SEALION: {str(e)}"

def initialize_models():
    """Initialize the embedding model and LLM."""
    try:
        # FIX 1: BYPASS LANGCHAIN FOR EMBEDDINGS
        from sentence_transformers import SentenceTransformer
        
        class DirectEmbeddings:
            def __init__(self):
                self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
            
            def embed_documents(self, texts):
                return self.model.encode(texts, convert_to_tensor=False).tolist()
            
            def embed_query(self, text):
                return self.model.encode(text, convert_to_tensor=False).tolist()
        
        embedding_model = DirectEmbeddings()
        logger.info("✅ Embedding model loaded successfully")
        
        # FIX 2: PROPERLY INITIALIZE SEALION
        api_key = os.getenv("SEALION_API_KEY")
        
        try:
            llm = SEALIONWrapper(api_key=api_key)
            logger.info("✅ SEALION LLM initialized")
            
            # Test the LLM connection
            test_response = llm._call("Hello")
            if "Error calling SEALION" in test_response:
                return None, f"❌ SEALION API test failed: {test_response}"
            
        except Exception as e:
            error_msg = f"❌ Error initializing SEALION: {str(e)}"
            logger.error(error_msg)
            return None, error_msg
        
        # Initialize GraphRAG
        try:
            graph_rag = GraphRAG(embedding_model, llm)
            logger.info("✅ GraphRAG system initialized")
        except Exception as e:
            error_msg = f"❌ Error initializing GraphRAG: {str(e)}"
            logger.error(error_msg)
            return None, error_msg
        
        return graph_rag, "✅ Models initialized successfully!"
        
    except ImportError as e:
        error_msg = f"❌ Missing dependency: {str(e)}. Run: pip install sentence-transformers"
        logger.error(error_msg)
        return None, error_msg
    except Exception as e:
        error_msg = f"❌ Unexpected error: {str(e)}"
        logger.error(error_msg)
        return None, error_msg

def process_pdf_files(files):
    """Process uploaded PDF files and build knowledge graph."""
    global graph_rag_system, documents_processed
    
    if not files:
        return "❌ No files uploaded. Please upload PDF files first."
    
    if graph_rag_system is None:
        graph_rag_system, init_msg = initialize_models()
        if graph_rag_system is None:
            return init_msg
    
    try:
        documents = []
        processed_files = []
        failed_files = []
        
        # Process each uploaded file
        for file_path in files:
            try:
                logger.info(f"Processing file: {file_path}")
                
                # Extract text from PDF
                reader = PdfReader(file_path)
                text = ""
                for page_num, page in enumerate(reader.pages):
                    try:
                        text += page.extract_text()
                    except Exception as e:
                        logger.warning(f"Error extracting page {page_num} from {file_path}: {e}")
                        continue
                
                if text.strip():
                    documents.append(Document(
                        page_content=text,
                        metadata={"source": os.path.basename(file_path)}
                    ))
                    processed_files.append(os.path.basename(file_path))
                    logger.info(f"✅ Successfully extracted text from {os.path.basename(file_path)}")
                else:
                    failed_files.append(os.path.basename(file_path))
                    logger.warning(f"❌ No text extracted from {os.path.basename(file_path)}")
                
            except Exception as e:
                failed_files.append(os.path.basename(file_path))
                logger.error(f"Error processing {file_path}: {e}")
                continue
        
        if not documents:
            return "❌ No valid PDF content found in uploaded files. Please check your PDFs contain extractable text."
        
        # Process documents through GraphRAG
        try:
            logger.info("Building knowledge graph...")
            graph_rag_system.process_documents(documents)
            documents_processed = True
            logger.info("✅ Knowledge graph built successfully")
        except Exception as e:
            error_msg = f"❌ Error building knowledge graph: {str(e)}"
            logger.error(error_msg)
            return error_msg
        
        # Get graph statistics
        try:
            num_nodes = len(graph_rag_system.knowledge_graph.graph.nodes())
            num_edges = len(graph_rag_system.knowledge_graph.graph.edges())
        except Exception as e:
            logger.warning(f"Error getting graph stats: {e}")
            num_nodes = num_edges = 0
        
        # Format success message
        success_msg = f"""✅ Successfully processed {len(documents)} documents!

📊 **Graph Statistics:**
- **Nodes:** {num_nodes}
- **Edges:** {num_edges}
- **Files processed:** {', '.join(processed_files)}"""
        
        if failed_files:
            success_msg += f"\n- **Failed files:** {', '.join(failed_files)}"
        
        success_msg += "\n\n💬 You can now start asking questions about your documents!"
        
        return success_msg
        
    except Exception as e:
        error_msg = f"❌ Unexpected error processing documents: {str(e)}"
        logger.error(error_msg)
        return error_msg

def chat_with_documents(message, history):
    """Handle chat interactions with the processed documents."""
    global graph_rag_system, documents_processed
    
    if not documents_processed or graph_rag_system is None:
        history.append(["Please upload and process PDF documents first.", ""])
        return history, ""
    
    if not message.strip():
        history.append(["Please enter a valid question.", ""])
        return history, ""
    
    try:
        logger.info(f"Processing query: {message}")
        
        # Process query through GraphRAG
        answer, traversal_path, context_data = graph_rag_system.query(message)
        
        # Format response with additional info
        response = f"{answer}"
        
        if traversal_path:
            response += f"\n\n📍 **Graph Analysis:** Traversed {len(traversal_path)} nodes in the knowledge graph to find this answer."
        
        if context_data and isinstance(context_data, dict):
            if context_data.get('num_sources', 0) > 0:
                response += f"\n📚 **Sources:** Used {context_data['num_sources']} document sections."
        
        history.append([message, response])
        logger.info("✅ Query processed successfully")
        return history, ""
        
    except Exception as e:
        error_msg = f"❌ Error processing your question: {str(e)}"
        logger.error(error_msg)
        history.append([message, error_msg])
        return history, ""

def clear_chat():
    """Clear the chat history."""
    return []

def reset_system():
    """Reset the entire system."""
    global graph_rag_system, documents_processed
    graph_rag_system = None
    documents_processed = False
    logger.info("🔄 System reset")
    return "🔄 System reset. Please upload documents again.", [], ""

def get_system_status():
    """Get current system status."""
    global graph_rag_system, documents_processed
    
    if graph_rag_system is None:
        return "❌ Models not initialized"
    elif not documents_processed:
        return "⚠️ Models ready, no documents processed"
    else:
        try:
            num_nodes = len(graph_rag_system.knowledge_graph.graph.nodes())
            num_edges = len(graph_rag_system.knowledge_graph.graph.edges())
            return f"✅ System ready - {num_nodes} nodes, {num_edges} edges"
        except Exception as e:
            return f"⚠️ System ready but error getting stats: {str(e)}"

def create_interface():
    """Create the Gradio interface."""
    with gr.Blocks(
        title="GraphRAG with SEALION",
        theme=gr.themes.Soft(),
        css="""
        .gradio-container {
            max-width: 1200px !important;
        }
        .header {
            text-align: center;
            margin-bottom: 30px;
        }
        .status-box {
            background-color: #f0f2f6;
            padding: 15px;
            border-radius: 10px;
            margin: 10px 0;
        }
        .chat-container {
            height: 400px;
        }
        """
    ) as demo:
        
        gr.HTML("""
        <div class="header">
            <h1>🤖 GraphRAG with SEALION LLM</h1>
            <p>Upload PDF documents and chat with them using Knowledge Graph-enhanced RAG</p>
        </div>
        """)
        
        with gr.Row():
            # Left column - Document Upload
            with gr.Column(scale=1):
                gr.HTML("<h2>📄 Document Upload</h2>")
                
                file_upload = gr.File(
                    label="Upload PDF Files",
                    file_types=[".pdf"],
                    file_count="multiple"
                )
                
                process_btn = gr.Button(
                    "🚀 Process Documents",
                    variant="primary",
                    size="lg"
                )
                
                upload_status = gr.Textbox(
                    label="Processing Status",
                    interactive=False,
                    lines=10,
                    value="Upload PDF files and click 'Process Documents' to begin..."
                )
                
                with gr.Row():
                    system_status = gr.Textbox(
                        label="System Status",
                        interactive=False,
                        value="❌ Models not initialized"
                    )
                    
                    refresh_status_btn = gr.Button("🔄 Refresh")
                    reset_btn = gr.Button("🗑️ Reset", variant="stop")
            
            # Right column - Chat Interface
            with gr.Column(scale=1):
                gr.HTML("<h2>💬 Chat with Documents</h2>")
                
                chatbot = gr.Chatbot(
                    label="Conversation",
                    height=400
                )
                
                with gr.Row():
                    msg_input = gr.Textbox(
                        label="Your Question",
                        scale=4
                    )
                    send_btn = gr.Button("Send", variant="primary", scale=1)
                
                with gr.Row():
                    clear_btn = gr.Button("🗑️ Clear Chat")
                
                gr.HTML("""
                <div class="status-box">
                    <h3>💡 Tips:</h3>
                    <ul>
                        <li>Upload multiple PDF files for richer knowledge graphs</li>
                        <li>Ask specific questions about content, relationships, or summaries</li>
                        <li>The system uses NetworkX graphs to find connections between concepts</li>
                        <li>Try questions like: "What are the main topics?" or "How are X and Y related?"</li>
                    </ul>
                </div>
                """)
        
        # Example questions
        with gr.Row():
            gr.HTML("<h3>🔍 Example Questions</h3>")
        
        example_questions = [
            "What are the main topics discussed in the documents?",
            "Can you summarize the key findings?",
            "How are the different concepts connected?",
            "What are the most important points mentioned?",
            "Are there any contradictions or conflicts in the information?"
        ]
        
        examples = gr.Examples(
            examples=[[q] for q in example_questions],
            inputs=[msg_input],
            label="Click on any example question:"
        )
        
        # Event handlers
        process_btn.click(
            fn=process_pdf_files,
            inputs=[file_upload],
            outputs=[upload_status],
            show_progress=True
        )
        
        def handle_send(message, history):
            new_history, empty_str = chat_with_documents(message, history)
            return new_history, ""
        
        send_btn.click(
            fn=handle_send,
            inputs=[msg_input, chatbot],
            outputs=[chatbot, msg_input],
            show_progress=True
        )
        
        msg_input.submit(
            fn=handle_send,
            inputs=[msg_input, chatbot],
            outputs=[chatbot, msg_input],
            show_progress=True
        )
        
        clear_btn.click(
            fn=clear_chat,
            outputs=[chatbot]
        )
        
        reset_btn.click(
            fn=reset_system,
            outputs=[upload_status, chatbot, system_status]
        )
        
        refresh_status_btn.click(
            fn=get_system_status,
            outputs=[system_status]
        )
        
        # Auto-refresh status on load
        demo.load(
            fn=get_system_status,
            outputs=[system_status]
        )
    
    return demo

if __name__ == "__main__":
    
    # Check for API key
    if not os.getenv("SEALION_API_KEY"):
        print("⚠️  Warning: SEALION_API_KEY not set!")
        print("Please set it with: export SEALION_API_KEY=your-api-key")
        print("Or uncomment and edit the line above in this file")
    
    # Create and launch the interface
    demo = create_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,  # Set to True if you want a public link
        show_error=True
    )