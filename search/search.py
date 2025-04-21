# search.py
from typing import List, Dict
import torch
import os
from dotenv import load_dotenv # If you use a .env file for config
# Updated import for Qdrant client:
from vectorstore.qdrant_client import get_qdrant_client
# Ensure ColPali classes are available (adjust path if needed)
from colpali_engine.models import ColPali, ColPaliProcessor
# Qdrant models for search parameters might be useful
from qdrant_client.http import models as rest

# --- Configuration ---
load_dotenv() # Loads environment variables from .env file if present

# Ensure these match the model used for embedding
MODEL_NAME = "vidore/colpali-v1.3"
COLPALI_PROCESSOR_NAME = "vidore/colpali-v1.3"

# Get collection name from env or use default (must match main.py)
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "my_pdf_collection")
# Search parameters
SEARCH_LIMIT = int(os.getenv("SEARCH_LIMIT", "5")) # Top K results
# --- End Configuration ---

# Global cache for model and processor to avoid reloading on every search call
# Be mindful of memory usage, especially with large models.
colpali_model = None
colpali_processor = None
compute_device = "cpu" # Default, will be updated in init

def init_colpali_resources():
    """Loads ColPali model and processor if not already loaded."""
    global colpali_model, colpali_processor, compute_device

    if colpali_model is not None and colpali_processor is not None:
        print("(Using cached ColPali model and processor)")
        return colpali_model, colpali_processor

    # Determine device dynamically
    if torch.cuda.is_available():
        compute_device = "cuda:0"
    elif torch.backends.mps.is_available(): # Check for Apple Silicon MPS
        compute_device = "mps"
    else:
        compute_device = "cpu"

    print(f"🧠 Loading ColPali model for search (using device: {compute_device})...")
    try:
        # Use float32 for CPU/MPS compatibility, bfloat16 potentially faster on compatible GPUs
        dtype = torch.bfloat16 if compute_device == "cuda:0" else torch.float32
        colpali_model = ColPali.from_pretrained(MODEL_NAME, torch_dtype=dtype, device_map=compute_device)
        colpali_processor = ColPaliProcessor.from_pretrained(COLPALI_PROCESSOR_NAME)
        print("✅ ColPali model and processor loaded for search.")
        return colpali_model, colpali_processor
    except Exception as e:
        print(f"❌ Failed to load ColPali resources for search: {e}")
        raise RuntimeError("Failed to initialize ColPali for search") from e


def process_query(query_text: str, processor, model) -> List[float]:
    """
    Converts query text into a single flat embedding using mean pooling.
    """
    with torch.no_grad():
        inputs = processor.process_queries([query_text]).to(model.device)
        output = model(**inputs)  # shape: [1, n_tokens, 128]
        print("Query embedding shape:", output.shape)

        # Mean pool across token axis (dim=1) → shape becomes [1, 128]
        pooled = output.mean(dim=1)

        # Flatten and convert to list of floats
        return pooled.squeeze(0).cpu().float().numpy().tolist()

def search_in_qdrant(query_vector: List[float], qdrant_client, collection_name: str, top_k: int = 5):
    """
    Queries Qdrant for the most similar vectors to the input query_vector.
    Returns a list of ScoredPoint objects.
    """
    print(f"🛰️ Searching Qdrant collection '{collection_name}' for top {top_k} results...")
    try:
        search_result = qdrant_client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=top_k,
            # Request payload to be returned with the results
            with_payload=True,
             # Optional: Add search parameters for tuning (e.g., HNSW parameters)
            # search_params=rest.SearchParams(hnsw_ef=128, exact=False)
        )
        print(f"✅ Qdrant search completed. Found {len(search_result)} potential matches.")
        # search_result is already a list of ScoredPoint objects
        return search_result
    except Exception as e:
        print(f"❌ Error during Qdrant search in collection '{collection_name}': {e}")
        raise RuntimeError("Qdrant search operation failed") from e

# Main function called by main.py
def search(query_text: str) -> List[Dict]:
    """
    Orchestrates the search process: initializes resources, embeds query, searches Qdrant, formats results.
    """
    # 1. Initialize Qdrant Client (connects on each search - consider caching if performance is critical)
    try:
        # Using the updated function from qdrant_client.py
        qdrant_client = get_qdrant_client()
    except Exception as e:
         print(f"❌ Search cannot proceed: Failed to connect to Qdrant: {e}")
         # Re-raise the exception so main.py can handle it or inform the user
         raise

    # 2. Initialize ColPali Model and Processor (uses cache)
    try:
        model, processor = init_colpali_resources()
    except Exception as e:
         print(f"❌ Search cannot proceed: Failed to initialize ColPali: {e}")
         raise

    # 3. Process the query to get its vector representation
    query_vector = process_query(query_text, processor, model)

    # 4. Search Qdrant for the most similar vectors
    search_results = search_in_qdrant(
        query_vector=query_vector,
        qdrant_client=qdrant_client,
        collection_name=COLLECTION_NAME,
        top_k=SEARCH_LIMIT
    )

    # 5. Format results into a list of dictionaries
    formatted_results = []
    if search_results:
        for i, point in enumerate(search_results):
            # point is a ScoredPoint object
            payload = point.payload if point.payload else {} # Ensure payload is a dict
            formatted_results.append({
                "rank": i + 1,
                "id": point.id,
                # Access payload directly - uses 'source' key set during embedding
                "source": payload.get("source", "Unknown"),
                "similarity": point.score # Cosine similarity score
            })

    return formatted_results

# Example standalone usage (optional)
if __name__ == '__main__':
    print("Testing search module independently...")
    test_query = "show me leafy plants"
    try:
        results = search(test_query)
        if results:
            print(f"\n--- Test Results for '{test_query}' ---")
            for res in results:
                print(f"Rank: {res['rank']}, Score: {res['similarity']:.4f}, Source: {os.path.basename(res['source'])}")
        else:
            print("No results found for test query.")
    except Exception as e:
        print(f"Test search failed: {e}")