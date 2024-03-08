from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def load_model_quantized(model_id, bit_count = None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if bit_count == 8:
        model = AutoModelForCausalLM.from_pretrained(model_id, device_map='auto', load_in_8bit=True)
    if bit_count == 16:
        model = AutoModelForCausalLM.from_pretrained(model_id, device_map='auto', torch_dtype=torch.float16)
    if bit_count == 32:
        model = AutoModelForCausalLM.from_pretrained(model_id, device_map='auto', torch_dtype=torch.float32)
    else:
        print("Bit count not valid, loading in int8 model")
        model = AutoModelForCausalLM.from_pretrained(model_id, device_map='auto', load_in_8bit=True)
    print(f"Model Size: {model.get_memory_footprint():,} bytes")
    return model, device

def main():
    model_id = "mistralai/Mistral-7B-Instruct-v0.2"
    
    bit_input = input("What bit size would you like to load in for weights? (8, 16, 32): ")
    

    model, device  = load_model_quantized(model_id, bit_input)
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