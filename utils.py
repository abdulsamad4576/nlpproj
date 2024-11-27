import time
from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    VectorParams,
    Distance,
    PointStruct
)
import os

from typing import List, Dict, Any

from dotenv import load_dotenv; load_dotenv()
load_dotenv()

from sentence_transformers import SentenceTransformer
import numpy as np


embedding_model = SentenceTransformer('all-mpnet-base-v2')  # Produces 1536-dimensional embeddings

def upscale_embedding(embedding: list, target_dim: int = 1536) -> list:
    """
    Upscales a lower-dimensional embedding to a higher dimension using interpolation.
    
    :param embedding: List of numbers (original embedding).
    :param target_dim: Target dimension (default: 1536).
    :return: Upscaled embedding as a list.
    """
    original_dim = len(embedding)
    upscaled_embedding = np.interp(
        np.linspace(0, original_dim, target_dim),  # Target indices
        np.arange(original_dim),                  # Original indices
        embedding                                 # Original embedding values
    )
    return upscaled_embedding.tolist()

def compute_time(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"\n>> Function '{func.__name__}' took {execution_time:.4f} seconds to execute.\n")
        return result
    return wrapper

# Connect to Qdrant
@compute_time
def connect_to_db(url: str, api_key: str) -> QdrantClient:
    client = QdrantClient(url=url, api_key=api_key)
    print(f"Connected to Qdrant at {url}")
    return client

# Create collection without payload schema using the new methods
@compute_time
def create_collection_if_not_exists(client: QdrantClient, collection_name: str, vector_size: int = 1536):
    if not client.collection_exists(collection_name=collection_name):
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
        )
        print(f"Collection {collection_name} created.")
    else:
        print(f"Collection {collection_name} already exists.")

# Add vectors using upsert
@compute_time
def add_vectors(client: QdrantClient, collection_name: str, vectors: List[PointStruct]):
    try:
        client.upsert(collection_name=collection_name, points=vectors)
    except Exception as e:
        print(f"Error during upsert: {e}")

@compute_time
def generate_vector(content: str) -> list:
    """
    Generates a 1536-dimensional embedding by first generating a 768-dimensional embedding
    and then upscaling it using interpolation.
    
    :param content: Input text to embed.
    :return: 1536-dimensional embedding as a list.
    """
    embedding = embedding_model.encode(content, show_progress_bar=False)  # 768-dimensional
    upscaled_embedding = upscale_embedding(embedding, target_dim=1536)  # Interpolate to 1536 dimensions
    return upscaled_embedding

def extract_step_ids(program_data: Dict[str, Any]) -> List[int]:
    """
    Extracts step_id as integers from program_data['steps'].
    """
    step_ids = []
    for step_data in program_data["steps"].values():
        step_id_str = step_data.get('step_id')
        if step_id_str:
            try:
                step_id = int(step_id_str)
                step_ids.append(step_id)
            except ValueError:
                print(f"Invalid step_id: {step_id_str}. Skipping.")
        else:
            print("No step_id found in step_data. Skipping.")
    return step_ids



