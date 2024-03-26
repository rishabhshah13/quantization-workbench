from transformers import AutoTokenizer
import torch
from awq import AutoAWQForCausalLM


def load_awq(model_id):
    # model_id = "TheBloke/Llama-2-7B-Chat-AWQ"
    # model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32,device_map='auto')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = AutoAWQForCausalLM.from_quantized(model_id, fuse_layers=True,
                                        trust_remote_code=False, safetensors=True)

    # model = AutoModelForCausalLM.from_pretrained("TheBloke/zephyr-7B-alpha-AWQ", attn_implementation="flash_attention_2", device_map='auto')
    # print(f"Model Size: {model.get_memory_footprint() / (1024**3):,} GB")
    # self.model = model

    return model, device



def load_tokenizer(model_id):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    return tokenizer