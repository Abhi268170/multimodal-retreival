import gradio as gr
import os
import shutil
import torch
from pathlib import Path
from PIL import Image
from loaders.pdf_loader import convert_pdf_to_images
from models.colpali_model import load_colpali_model
from embedders.image_embedder import embed_images
from vectorstore.qdrant_client import get_qdrant_client, create_collection, upsert_embeddings
from search.search import search
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
IMAGE_DIR = os.getenv("IMAGE_DIR", "images")
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "my_pdf_collection")
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "1"))
COMPUTE_DEVICE = "cuda:0" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

# Global variables to store model and processor
model = None
processor = None

def init_models():
    """Initialize the ColPali model and processor"""
    global model, processor
    if model is None or processor is None:
        model, processor = load_colpali_model(device=COMPUTE_DEVICE)
    return model, processor

def process_pdf(pdf_file):
    """Process uploaded PDF and store embeddings"""
    try:
        # Save uploaded PDF temporarily
        pdf_path = Path("temp.pdf")
        shutil.copy(pdf_file.name, pdf_path)

        # Convert PDF to images
        image_paths = convert_pdf_to_images(str(pdf_path), IMAGE_DIR)
        if not image_paths:
            return "Error: No pages were extracted from the PDF."

        # Initialize model and get embeddings
        model, processor = init_models()
        embeddings = embed_images(
            image_paths=image_paths,
            model=model,
            processor=processor,
            device=COMPUTE_DEVICE,
            batch_size=BATCH_SIZE
        )

        # Store embeddings in Qdrant
        qdrant_client = get_qdrant_client()
        create_collection(qdrant_client, COLLECTION_NAME)
        upsert_embeddings(qdrant_client, COLLECTION_NAME, embeddings)

        # Cleanup temporary PDF
        pdf_path.unlink()

        return f"✅ Successfully processed PDF with {len(image_paths)} pages. Ready for searching!"
    except Exception as e:
        return f"❌ Error processing PDF: {str(e)}"

def search_pdf(query):
    """Search through the processed PDF pages"""
    try:
        if not query.strip():
            return "Please enter a search query."

        results = search(query)
        if not results:
            return "No matching pages found."

        # Format results and load images
        output = []
        for res in results:
            source_path = res.get('source', '')
            if os.path.exists(source_path):
                img = Image.open(source_path)
                similarity = res.get('similarity', 0.0)
                output.append((img, f"Similarity: {similarity:.4f}"))

        return output
    except Exception as e:
        return f"❌ Error during search: {str(e)}"

# Create Gradio interface
with gr.Blocks(title="PDF Visual Search") as demo:
    gr.Markdown("# PDF Visual Search")
    gr.Markdown("Upload a PDF and search through its pages using natural language.")
    
    with gr.Row():
        with gr.Column():
            pdf_input = gr.File(label="Upload PDF")
            process_btn = gr.Button("Process PDF")
            status_output = gr.Textbox(label="Status")
        
    with gr.Row():
        with gr.Column():
            query_input = gr.Textbox(label="Search Query", placeholder="Enter your search query...")
            search_btn = gr.Button("Search")
        
    gallery = gr.Gallery(label="Search Results", columns=3, height=400)

    # Set up event handlers
    process_btn.click(
        fn=process_pdf,
        inputs=[pdf_input],
        outputs=[status_output]
    )
    
    search_btn.click(
        fn=search_pdf,
        inputs=[query_input],
        outputs=[gallery]
    )

if __name__ == "__main__":
    # Ensure the image directory exists
    os.makedirs(IMAGE_DIR, exist_ok=True)
    # Launch the interface
    demo.launch()