from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
from transformers import BitsAndBytesConfig
from torch import bfloat16

def load_model_quantized(model_id, quantized = True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map='auto', load_in_8bit=quantized)
    print(f"Model Size: {model.get_memory_footprint():,} bytes")
    return model, device

def main():
    model_id = "mistralai/Mistral-7B-Instruct-v0.2"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Prompt user for quantized model usage
#     quantized_input = input("Do you want to use the quantized version of the model? (yes/no): ").strip().lower()
#     if quantized_input in {"yes", "y"}:
#         quantized = True
#     elif quantized_input in {"no", "n"}:
#         quantized = False
#     else:
#         print("Invalid input. Please type 'yes' or 'no'.")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, #4-bit quantization
        bnb_4bit_quant_type='nf4', #Normalized float 4
        bnb_4bit_use_double_quant=True, #Second quantization after the first
        bnb_4bit_compute_dtype=bfloat16 #Computation type
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map='auto'
    )
    
    if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
    pipe = pipeline(model=model, tokenizer=tokenizer, task='text-generation')
    
    while True:
        # User input
        user_input = input("You: ")
        if not user_input.strip():
            print("Exiting...")
            break

        outputs = pipe(
            user_input,
            max_new_tokens=1024,
            do_sample=True,
            temperature=0.1,
            top_p=0.95
        )
        print("Model: ", outputs[0]["generated_text"])
if __name__ == "__main__":
    main()