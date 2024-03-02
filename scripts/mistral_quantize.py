from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def load_model_quantized(model_id, quantized = True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map='auto', load_in_8bit=quantized)
    print(f"Model Size: {model.get_memory_footprint():,} bytes")
    return model, device

def main():
    model_id = "mistralai/Mistral-7B-Instruct-v0.2"
    
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
        
        generated_ids = model.generate(model_inputs, max_new_tokens = 1000, do_sample=True)
        decoded = tokenizer.batch_device(generated_ids)
        print("Model: ", decoded[0])

if __name__ == "__main__":
    main()