from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def load_model_quantized(model_id, quantized = True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map='auto', load_in_8bit=quantized)
    print(f"Model Size: {model.get_memory_footprint():,} bytes")
    return model, device

def main():
    model_id = "mistralai/Mistral-7B-v0.1"
    
    # Prompt user for quantized model usage
    quantized_input = input("Do you want to use the quantized version of the model? (yes/no): ").strip().lower()
    if quantized_input in {"yes", "y"}:
        quantized = True
    elif quantized_input in {"no", "n"}:
        quantized = False
    else:
        print("Invalid input. Please type 'yes' or 'no'.")

    model, device  = load_model_quantized(model_id, quantized)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

    while True:
        # User input
        user_input = input("You: ")
        if not user_input.strip():
            print("Exiting...")
            break

        # Encode user input directly
        encoded_input = tokenizer(user_input, return_tensors="pt").to(device)
        
        # Generate response
        generated_ids = model.generate(encoded_input, pad_token_id=tokenizer.pad_token_id, max_length=1000, do_sample=True)
        decoded = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        print("Model:", decoded[0])


if __name__ == "__main__":
    main()