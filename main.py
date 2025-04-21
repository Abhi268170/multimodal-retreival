# main.py
import os
import sys
from loaders.pdf_loader import convert_pdf_to_images
from models.colpali_model import load_colpali_model as load_colpali
from embedders.image_embedder import embed_images
# Updated import:
from vectorstore.qdrant_client import (
    get_qdrant_client,
    create_collection,
    upsert_embeddings
)
from search.search import search
# Optional: Load environment variables from .env file
# from dotenv import load_dotenv
# load_dotenv()

# --- Config ---
PDF_FILE = os.getenv("PDF_FILE", "data/plants.pdf") # Allow overriding via env var
IMAGE_DIR = os.getenv("IMAGE_DIR", "images")
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "my_pdf_collection")
# Consider increasing batch size if hardware allows (GPU/RAM) for faster embedding
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "1"))
# Determine compute device ('cuda:0', 'mps', or 'cpu')
# You might want to add logic to select automatically based on torch.cuda.is_available() etc.
COMPUTE_DEVICE = os.getenv("COMPUTE_DEVICE", "cpu")
# --- End Config ---

def main():
    # 0. Validate PDF file exists
    if not os.path.exists(PDF_FILE):
        print(f"❌ Error: PDF file not found at '{PDF_FILE}'")
        sys.exit(1)

    # 1. Convert PDF pages to image files
    print(f"📄 Converting PDF '{PDF_FILE}' pages to images in '{IMAGE_DIR}'...")
    try:
        image_paths = convert_pdf_to_images(PDF_FILE, IMAGE_DIR)
        if not image_paths:
            print("⚠️ No pages were converted from the PDF.")
            # Decide if this is an error or just an empty PDF
            sys.exit(1) # Exit if no images generated
        print(f"✅ Converted {len(image_paths)} pages to images.")
    except Exception as e:
        print(f"❌ Failed during PDF conversion: {e}")
        sys.exit(1)

    # 2. Load ColPali model & processor
    print(f"🧠 Loading ColPali model (using device: {COMPUTE_DEVICE})...")
    try:
        model, processor = load_colpali(device=COMPUTE_DEVICE)
    except Exception as e:
        print(f"❌ Failed to load ColPali model: {e}")
        print("   Ensure 'colpali-engine' and its dependencies (like transformers, torch) are installed.")
        sys.exit(1)

    # 3. Embed images
    print(f"🧬 Generating embeddings for {len(image_paths)} images (batch size: {BATCH_SIZE})...")
    try:
        embeddings = embed_images(
            image_paths=image_paths,
            model=model,
            processor=processor,
            device=COMPUTE_DEVICE,
            batch_size=BATCH_SIZE
        )
        if not embeddings:
             print("❌ No embeddings were generated. Exiting.")
             sys.exit(1)
        print(f"✅ Generated {len(embeddings)} embeddings.")
    except Exception as e:
        print(f"❌ Failed during image embedding: {e}")
        sys.exit(1)

    # 4. Setup Qdrant client
    try:
        qdrant_client = get_qdrant_client() # Connect to Qdrant
    except Exception as e:
        print(f"❌ Failed to connect to Qdrant: {e}")
        print("   Ensure the Qdrant server is running and accessible at the configured URL.")
        sys.exit(1) # Exit if connection fails

    # 5. Create or verify Qdrant collection
    try:
         create_collection(qdrant_client, COLLECTION_NAME) # Pass the client object
    except Exception as e:
         print(f"❌ Halting due to error during collection setup: {e}")
         sys.exit(1) # Exit if collection setup fails

    # 6. Upsert embeddings into Qdrant
    try:
        upsert_embeddings(qdrant_client, COLLECTION_NAME, embeddings) # Pass client object
    except Exception as e:
        print(f"❌ Failed during vector upload: {e}")
        sys.exit(1) # Exit if upload fails

    # 7. Search loop
    print("\n==========================================")
    print("✅ Setup complete.")
    print(f"🔍 Ready to search PDF '{os.path.basename(PDF_FILE)}' stored in collection '{COLLECTION_NAME}'.")
    print("==========================================")
    while True:
        try:
            query = input("Enter your query (or type 'exit' to quit): ")
        except EOFError: # Handle Ctrl+D
            print("\nExiting.")
            break

        query = query.strip()
        if query.lower() == "exit":
            print("Exiting.")
            break
        if not query:
            # print("⚠️ Please enter a query.") # Optional: prompt again
            continue

        print(f"🔍 Searching for: '{query}'...")
        try:
            results = search(query) # Call the search function from search.py
            if results:
                print(f"🎯 Found {len(results)} results:")
                for i, res in enumerate(results, 1):
                    source_path = res.get('source', 'Unknown')
                    # Extract just the filename (page identifier)
                    page_identifier = os.path.basename(source_path) if source_path != 'Unknown' else 'Unknown'
                    similarity = res.get('similarity', float('nan')) # Use NaN if missing

                    print(f"\n--- Result {i} ---")
                    print(f"  Similarity: {similarity:.4f}")
                    print(f"  Source Page: {page_identifier}")
                    # Optionally print the full path or other metadata if needed
                    # print(f"  ID: {res.get('id', 'N/A')}")
            else:
                print("🤷 No relevant pages found for your query.")
        except Exception as e:
            print(f"❌ An error occurred during search: {e}")
            # Consider logging the full traceback for debugging
            # import traceback
            # traceback.print_exc()

        print("\n" + "—" * 40 + "\n") # Separator for next query

if __name__ == "__main__":
    main()