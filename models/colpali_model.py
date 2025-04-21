import torch
from colpali_engine.models import ColPali, ColPaliProcessor

# You can customize this if you upgrade the model
MODEL_NAME = "vidore/colpali-v1.3"

def load_colpali_model(device: str = "cpu"):
    """
    Load the ColPali model and processor.

    Args:
        device (str): 'cpu', 'cuda:0' or 'mps' depending on your hardware.

    Returns:
        Tuple of (model, processor)
    """
    model = ColPali.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16 if device != "cpu" else torch.float32,
        device_map=device,
    )
    processor = ColPaliProcessor.from_pretrained(MODEL_NAME)

    return model, processor
