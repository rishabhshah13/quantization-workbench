from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
import os
from huggingface_hub import login
# from vllm import LLM, SamplingParams

class mModel:

    def __init__(self,model_name,quantization):

        self.model_name = model_name
        self.quantization = quantization
        self.model_id = "mistralai/Mistral-7B-Instruct-v0.2"
        self.tokenizer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        print(f"Initializing Model: {self.model_name}")
    
    def load_awq(self):
        model_id = "TheBloke/zephyr-7B-alpha-AWQ"
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32,device_map='auto')
        # model = AutoModelForCausalLM.from_pretrained("TheBloke/zephyr-7B-alpha-AWQ", attn_implementation="flash_attention_2", device_map='auto')
        print(f"Model Size: {model.get_memory_footprint() / (1024**3):,} GB")
        self.model = model

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
    

    def load_vllm(self):
        # model_id = "TheBloke/zephyr-7B-alpha-AWQ"
        # model = LLM(model=model_id, quantization="awq", dtype="half")
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM

        model_id = "tiiuae/falcon-7b"
        tokenizer = AutoTokenizer.from_pretrained(model_id)

        model = AutoModelForCausalLM.from_pretrained(
            model_id, 
            torch_dtype=torch.bfloat16, 
            attn_implementation="flash_attention_2",
        )
        self.model = model



        # model = AutoModelForCausalLM.from_pretrained("TheBloke/zephyr-7B-alpha-AWQ", attn_implementation="flash_attention_2", device_map='auto')
        print(f"Model Size: {model.get_memory_footprint() / (1024**3):,} bytes")
        self.model = model


    def _load_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
    

    def start_inference(self):

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
            encoded_string = self.tokenizer.apply_chat_template(messages, return_tensors="pt")

            model_inputs = encoded_string.to(self.device)
            
            generated_ids = self.model.generate(model_inputs, pad_token_id=self.tokenizer.pad_token_id, max_new_tokens = 1000, do_sample=True)
            decoded = self.tokenizer.batch_decode(generated_ids)
            print("Model: ", decoded[0])    
    
    
    def single_inference(self,text):
        
        self._load_tokenizer()

        messages = [
            {"role": "user", "content": text}
        ]
        encoded_string = self.tokenizer.apply_chat_template(messages, return_tensors="pt")

        model_inputs = encoded_string.to(self.model.device)
        
        generated_ids = self.model.generate(model_inputs, pad_token_id=self.tokenizer.pad_token_id, max_new_tokens = 1000, do_sample=True)
        decoded = self.tokenizer.batch_decode(generated_ids)
        print("Model: ", decoded[0])    
    
    
    