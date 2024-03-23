from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
import os
from huggingface_hub import login

class mModel:

    def __init__(self,model_name,quantization):

        self.model_name = model_name
        self.quantization = quantization
        self.model_id = "mistralai/Mistral-7B-Instruct-v0.2"
        self.tokenizer = None
        self.device = 'cpu'
        
    
    def load(self):
        # load model in this one
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Bits Selected ", self.quantization)
        if self.quantization == 4:
            nf4_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16
            )
            model = AutoModelForCausalLM.from_pretrained(self.model_id, device_map='mps', quantization_config = nf4_config)
        if self.quantization == 8:
            nf8_config = BitsAndBytesConfig(
                load_in_8bit=True
            )
            model = AutoModelForCausalLM.from_pretrained(self.model_id, device_map='auto', quantization_config=nf8_config)
        if self.quantization == 16:
            model = AutoModelForCausalLM.from_pretrained(self.model_id, device_map='auto', torch_dtype=torch.float16)
        if self.quantization == 32:
            model = AutoModelForCausalLM.from_pretrained(self.model_id, device_map='auto')
        print(f"Model Size: {model.get_memory_footprint():,} bytes")
        return model

    def _load_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
    


    def start_inference(self):
        # input text and get output
        
        self._load_tokenizer()

        while True:
            # User input
            user_input = input("You: ")
            if not user_input.strip():
                print("Exiting...")
                break

            messages = [
                {"role": "user", "content": user_input}
            ]
            encodeds = self.tokenizer.apply_chat_template(messages, return_tensors="pt")

            model_inputs = encodeds.to(self.device)
            
            generated_ids = self.model.generate(model_inputs, pad_token_id=self.tokenizer.pad_token_id, max_new_tokens = 1000, do_sample=True)
            decoded = self.tokenizer.batch_decode(generated_ids)
            print("Model: ", decoded[0])