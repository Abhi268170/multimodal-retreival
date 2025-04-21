import torch
import uuid
from PIL import Image
from typing import List, Dict

def resize_image(image, size=(384, 384)):
    """
    Resize the input image to the specified size.
    """
    return image.resize(size, Image.Resampling.LANCZOS)

def embed_images(
    image_paths: List[str],
    model,
    processor,
    device: str = "cpu",
    batch_size: int = 1
) -> List[Dict]:
    """
    Generates vector embeddings from a list of image file paths using ColPali.

    Args:
        image_paths (List[str]): List of file paths to images.
        model: ColPali model.
        processor: ColPaliProcessor.
        device (str): 'cpu' or 'cuda:0'.
        batch_size (int): Batch size for inference.

    Returns:
        List[Dict]: A list of dictionaries, each containing:
            - id: unique id
            - vector: list of floats (the embedding)
            - metadata: a dict with the source info (like file name)
    """
    model.to(device)
    results = []

    print(f"🧬 Generating embeddings for {len(image_paths)} images (batch size: {batch_size})...")

    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i + batch_size]
        batch_images = [resize_image(Image.open(path).convert("RGB")) for path in batch_paths]

        with torch.no_grad():
            inputs = processor.process_images(batch_images).to(device)
            embeddings = model(**inputs)  # Expecting shape: [batch_size, 1, 128] or [batch_size, 128]

        for path, embedding in zip(batch_paths, embeddings):
            # Flatten the embedding in case it's shaped [1, 128]
            flat_vector = embedding[0].cpu().float().numpy().tolist() if embedding.ndim > 1 else embedding.cpu().float().numpy().tolist()

            results.append({
                "id": uuid.uuid4().hex,
                "vector": flat_vector,
                "metadata": {
                    "source": path
                }
            })

    if results:
        print("Sample embedding:", torch.tensor(results[0]['vector']).unsqueeze(0))

    print(f"✅ Generated {len(results)} embeddings.")
    return results
