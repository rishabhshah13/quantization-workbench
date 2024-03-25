# model.py

from scripts.mistral_quantize import load_model_quantized
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
import os
from huggingface_hub import login
import traceback

class Model:
    def __init__(self, model_name, bit_count):
        self.model, self.device = load_model_quantized(model_name, bit_count)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.context = []

    def get_output(self, user_input):
        self.context.append({"role": "user", "content": user_input})
        encodeds = self.tokenizer.apply_chat_template(self.context, return_tensors="pt")
        model_inputs = encodeds.to(self.device)
        generated_ids = self.model.generate(model_inputs, pad_token_id=self.tokenizer.pad_token_id, max_new_tokens = 1000, do_sample=True)
        decoded = self.tokenizer.batch_decode(generated_ids)
        self.context.append({"role": "assistant", "content": decoded[0]})
        return decoded[0]
