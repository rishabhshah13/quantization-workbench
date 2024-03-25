from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
import os
from huggingface_hub import login

class Model:

    def __init__(self,model_name,quantization):

        self.model_name = model_name
        self.quantization = quantization
        self.model_id = "mistralai/Mistral-7B-Instruct-v0.2"
        self.tokenizer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        print(f"Initializing Model: {self.model_name}")
    

    def load_bnb(self):
        # load model in this one
        print("Bits Selected ", self.quantization)
        if self.quantization == 4:
            nf4_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16
            )
            model = AutoModelForCausalLM.from_pretrained(self.model_id, device_map='auto', quantization_config = nf4_config)
        if self.quantization == 8:
            nf8_config = BitsAndBytesConfig(
                load_in_8bit=True
            )
            model = AutoModelForCausalLM.from_pretrained(self.model_id, device_map='auto', quantization_config=nf8_config)
        if self.quantization == 16:
            model = AutoModelForCausalLM.from_pretrained(self.model_id, device_map='auto', torch_dtype=torch.float16)
        if self.quantization == 32:
            model = AutoModelForCausalLM.from_pretrained(self.model_id, device_map='auto')

        print(f"Model Size: {model.get_memory_footprint() / (1024**3) :,} GB")

        self.model = model
    
        # return model
    
    def _load_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
    
    