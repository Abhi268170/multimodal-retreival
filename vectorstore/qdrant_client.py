# vectorstore/qdrant_client.py
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import Distance, VectorParams, PointStruct, CollectionStatus
# Use this for catching specific qdrant exceptions if needed
# from qdrant_client.http.exceptions import UnexpectedResponseError
import uuid
import os
from typing import List, Dict

# --- Configuration ---
# Fetch Qdrant URL from environment variable or use default
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")

# !! IMPORTANT !! Verify this size matches your ColPali model's output dimension.
# The comment in image_embedder.py suggests 128.
VECTOR_SIZE = 128
DISTANCE_METRIC = Distance.COSINE
# --- End Configuration ---

def get_qdrant_client() -> QdrantClient:
    """
    Initializes and returns a Qdrant client connected to the specified URL.
    Raises an exception if the connection fails.
    """
    print(f"🔌 Attempting to connect to Qdrant at {QDRANT_URL}...")
    client = QdrantClient(url=QDRANT_URL, timeout=60) # Increased timeout

    # Simple check to confirm connectivity
    try:
        # Attempt to get cluster info as a connection test
        _ = client.get_collections()
        print("✅ Successfully connected to Qdrant.")
        return client
    except Exception as e:
        print(f"❌ Failed to connect to Qdrant or verify connection: {e}")
        raise ConnectionError(f"Could not connect to Qdrant at {QDRANT_URL}") from e

def create_collection(client, collection_name: str, vector_size: int = 128):
    try:
        if client.get_collection(collection_name):
            print(f"⚠️ Collection '{collection_name}' already exists. Skipping creation.")
            return
    except Exception:
        print(f"Collection '{collection_name}' not found or error checking. Attempting creation...")

    print(f"🏗️ Creating collection '{collection_name}' with vector size {vector_size} and distance Cosine...")

    try:
        client.recreate_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=vector_size,
                distance=Distance.COSINE
            )
        )
        print(f"✅ Successfully created collection '{collection_name}'.")
    except Exception as e:
        print(f"❌ Failed to create collection '{collection_name}': {e}")
        raise   

def upsert_embeddings(client: QdrantClient, collection_name: str, embeddings: List[Dict]):
    """
    Uploads (Upserts) a list of embeddings into the specified Qdrant collection.
    Each embedding should be a dict with keys: 'id', 'vector', 'metadata'.
    """
    if not embeddings:
        print("⚠️ No embeddings provided to upsert.")
        return

    points_to_upsert = [
        PointStruct(
            id=item["id"], # Using the UUID generated in image_embedder
            vector=item["vector"],
            payload=item.get("metadata", {}) # Use the actual metadata dict from embedder
        )
        for item in embeddings
    ]

    print(f"📤 Upserting {len(points_to_upsert)} vectors into '{collection_name}'...")
    try:
        # Using upsert operation
        client.upsert(
            collection_name=collection_name,
            points=points_to_upsert,
            wait=True # Wait for the operation to complete for consistency
        )
        print(f"✅ Vectors upserted successfully into '{collection_name}'.")
    except Exception as e:
        print(f"❌ Failed to upsert vectors into '{collection_name}': {e}")
        # Decide how to handle failure
        raise RuntimeError(f"Failed to upsert vectors into Qdrant collection '{collection_name}'") from e

# Ensure the old init_qdrant function is removed or renamed if needed for other purposes.
# Ensure the line `get_qdrant_client = init_qdrant` is removed.