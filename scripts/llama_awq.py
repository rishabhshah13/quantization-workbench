from transformers import AutoTokenizer
import torch
from awq import AutoAWQForCausalLM


def load_awq(model_id):
    """
    Loads an AWQ (Adaptive Weights Quantization) model for causal language modeling.

    Args:
        model_id (str): The identifier for the AWQ model to load.

    Returns:
        tuple: A tuple containing the loaded model and the device it's on.
    """
    # Determine the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the AWQ model
    model = AutoAWQForCausalLM.from_quantized(model_id, fuse_layers=True,
                                              trust_remote_code=False, safetensors=True)

    return model, device


def load_tokenizer(model_id):
    """
    Loads the tokenizer for the given model.

    Args:
        model_id (str): The identifier for the model.

    Returns:
        PreTrainedTokenizer: The loaded tokenizer.
    """
    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer
