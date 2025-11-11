"""
Gradio App Interface for Task-Oriented RAG Chatbot
"""

import gradio as gr
from retriever import Retriever
from llm_handler import LLMHandler, SimpleLLMHandler
from typing import Tuple, List
import os


class RAGChatbot:
    """Task-Oriented RAG Chatbot Application"""
    
    def __init__(self, use_simple_model: bool = False):
        """
        Initialize the chatbot
        
        Args:
            use_simple_model: Use SimpleLLMHandler instead of full Mistral 7B
        """
        print("Initializing RAG Chatbot...")
        
        # Initialize retriever
        print("\n1. Loading retriever...")
        self.retriever = Retriever(top_k=5)
        
        # Initialize LLM
        print("\n2. Loading LLM...")
        if use_simple_model:
            self.llm = SimpleLLMHandler()
        else:
            self.llm = LLMHandler(load_in_4bit=True)
        
        print("\n✓ Chatbot initialized successfully!")
    
    def answer_query(
        self,
        query: str,
        history: List[Tuple[str, str]] = None
    ) -> Tuple[str, str]:
        """
        Process user query and return response with sources
        
        Args:
            query: User question
            history: Chat history (for Gradio chatbot interface)
            
        Returns:
            Tuple of (response, sources_text)
        """
        if not query.strip():
            return "Please enter a question.", ""
        
        # Retrieve relevant documents
        retrieval_result = self.retriever.retrieve_and_format(query, top_k=5)
        
        # Generate response
        if isinstance(self.llm, SimpleLLMHandler):
            response = self.llm.generate_response(query, retrieval_result['context'])
        else:
            result = self.llm.generate_with_retrieval(
                query=query,
                retrieval_result=retrieval_result,
                max_new_tokens=512
            )
            response = result['response']
        
        # Format sources
        sources = retrieval_result['sources']
        sources_text = self._format_sources(sources)
        
        return response, sources_text
    
    def _format_sources(self, sources: List[dict]) -> str:
        """Format sources for display"""
        if not sources:
            return "No sources found."
        
        formatted = "### Retrieved Sources:\n\n"
        for i, source in enumerate(sources, 1):
            formatted += f"**{i}. {source['title']}**\n"
            formatted += f"- Category: {source['category']}\n"
            formatted += f"- Relevance Score: {source['score']}\n"
            if source.get('url'):
                formatted += f"- [View Guide]({source['url']})\n"
            formatted += "\n"
        
        return formatted
    
    def create_interface(self) -> gr.Blocks:
        """Create Gradio interface"""
        
        with gr.Blocks(title="Task-Oriented Repair Assistant", theme=gr.themes.Soft()) as interface:
            gr.Markdown(
                """
                # 🔧 Task-Oriented Repair Assistant
                
                Ask me anything about device repair, troubleshooting, or technical tasks!
                I'll provide step-by-step guidance based on expert repair guides.
                
                **Example questions:**
                - How do I replace an iPhone screen?
                - My laptop won't turn on, what should I check?
                - How to fix a broken headphone jack?
                - Steps to replace a phone battery
                """
            )
            
            with gr.Row():
                with gr.Column(scale=2):
                    # Input section
                    query_input = gr.Textbox(
                        label="Your Question",
                        placeholder="e.g., How do I replace my iPhone battery?",
                        lines=3
                    )
                    
                    with gr.Row():
                        submit_btn = gr.Button("Get Answer", variant="primary", size="lg")
                        clear_btn = gr.Button("Clear", size="lg")
                    
                    # Output section
                    gr.Markdown("## 📝 Answer")
                    response_output = gr.Textbox(
                        label="Step-by-Step Instructions",
                        lines=10,
                        show_label=False
                    )
                
                with gr.Column(scale=1):
                    gr.Markdown("## 📚 Sources")
                    sources_output = gr.Markdown(
                        value="Sources will appear here after you submit a question."
                    )
            
            # Examples
            gr.Examples(
                examples=[
                    "How do I replace an iPhone screen?",
                    "My laptop won't turn on. What should I check?",
                    "How to replace a phone battery safely?",
                    "Fix broken headphone jack",
                    "How to clean laptop keyboard?",
                    "Replace graphics card thermal paste"
                ],
                inputs=query_input,
                label="Example Questions"
            )
            
            # Footer
            gr.Markdown(
                """
                ---
                💡 **Tip:** Be specific in your questions for better results!
                
                Data source: [MyFixit Dataset](https://github.com/microsoft/MyFixit-Dataset) - iFixit repair guides
                """
            )
            
            # Event handlers
            def process_query(query):
                response, sources = self.answer_query(query)
                return response, sources
            
            submit_btn.click(
                fn=process_query,
                inputs=[query_input],
                outputs=[response_output, sources_output]
            )
            
            clear_btn.click(
                fn=lambda: ("", "", "Sources will appear here after you submit a question."),
                outputs=[query_input, response_output, sources_output]
            )
            
            # Enter key support
            query_input.submit(
                fn=process_query,
                inputs=[query_input],
                outputs=[response_output, sources_output]
            )
        
        return interface
    
    def launch(self, share: bool = False, server_name: str = "127.0.0.1", server_port: int = 7860):
        """
        Launch the Gradio interface
        
        Args:
            share: Create public link (useful for Colab)
            server_name: Server address
            server_port: Server port
        """
        interface = self.create_interface()
        
        interface.launch(
            share=share,
            server_name=server_name,
            server_port=server_port,
            show_error=True
        )


def main():
    """Launch the chatbot application"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Task-Oriented RAG Chatbot")
    parser.add_argument(
        "--simple",
        action="store_true",
        help="Use simplified LLM model (faster, less accurate)"
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create public Gradio link"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Server port (default: 7860)"
    )
    
    args = parser.parse_args()
    
    # Check if index exists
    if not os.path.exists("faiss_index.bin"):
        print("="*80)
        print("ERROR: FAISS index not found!")
        print("="*80)
        print("\nPlease run the following steps first:")
        print("1. python data_processor.py  # Process documents")
        print("2. python embeddings.py      # Create embeddings and index")
        print("3. python app.py             # Launch chatbot")
        print("\nOr use: python main.py --build  # To run all steps automatically")
        return
    
    # Initialize and launch chatbot
    chatbot = RAGChatbot(use_simple_model=args.simple)
    
    print("\n" + "="*80)
    print("LAUNCHING GRADIO INTERFACE")
    print("="*80)
    
    chatbot.launch(
        share=args.share,
        server_port=args.port
    )


if __name__ == "__main__":
    main()
