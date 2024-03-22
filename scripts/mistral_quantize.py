from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
import os
from huggingface_hub import login

def load_model_quantized(model_id, bit_count = None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Bits Selected ", bit_count)
    if bit_count == 4:
        nf4_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        model = AutoModelForCausalLM.from_pretrained(model_id, device_map='auto', quantization_config = nf4_config)
    if bit_count == 8:
        nf8_config = BitsAndBytesConfig(
            load_in_8bit=True
        )
        model = AutoModelForCausalLM.from_pretrained(model_id, device_map='auto', quantization_config=nf8_config)
    if bit_count == 16:
        model = AutoModelForCausalLM.from_pretrained(model_id, device_map='auto', torch_dtype=torch.float16)
    if bit_count == 32:
        model = AutoModelForCausalLM.from_pretrained(model_id, device_map='auto', torch_dtype=torch.float32)
    print(f"Model Size: {model.get_memory_footprint():,} bytes")
    return model, device

def main():
    model_id = "mistralai/Mistral-7B-Instruct-v0.2"
    
    bit_input = int(input("Enter the number of bits (4, 8, 16, or 32): "))
    if bit_input not in [4, 8, 16, 32]:
        print("Invalid bit count. Loading model in int8 (8-bit) mode by default.")
        bit_input = 8

    model, device  = load_model_quantized(model_id, bit_input)

    model_key = f"{model_id}-{bit_input}-bit-quantized"
    if os.path.isfile("saved_models.txt"):
        with open("saved_models.txt", "r+") as file:
            if model_key in file.read():
                print("Model already saved to HuggingFace")
            else:
                login()
                model.push_to_hub(model_key)
                file.write(model_key+ "\n")
                print(f"{model_key} added to saved_models.txt")


    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

    while True:
        # User input
        user_input = input("You: ")
        if not user_input.strip():
            print("Exiting...")
            break

        messages = [
            {"role": "user", "content": user_input}
        ]

        encodeds = tokenizer.apply_chat_template(messages, return_tensors="pt")

        model_inputs = encodeds.to(device)
        
        generated_ids = model.generate(model_inputs, pad_token_id=tokenizer.pad_token_id, max_new_tokens = 1000, do_sample=True)
        decoded = tokenizer.batch_decode(generated_ids)
        print("Model: ", decoded[0])

if __name__ == "__main__":
    main()