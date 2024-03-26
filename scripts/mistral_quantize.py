from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
import os
from huggingface_hub import login

def load_model_quantized(model_id, bit_count = None):
    """
    Loads a quantized model from the Hugging Face Hub or from a local cache if available.

    Args:
        model_id (str): The identifier for the model to load.
        bit_count (int, optional): Number of bits for quantization. Defaults to None.

    Returns:
        tuple: A tuple containing the loaded model and the device it's on.
    """
    # Determine the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Bits Selected ", bit_count)
    # Check if model is already saved
    model_presaved = False
    model_key = f"{model_id}-{bit_count}-bit-quantized"
    model_key = model_key[model_key.rfind('/') + 1:]
    if os.path.isfile("saved_models.txt"):
        with open("saved_models.txt", "r+") as file:
            if model_key in file.read():
                model_presaved = True
                print("Loading quantized mo HuggingFace")
            else:
                # If not saved, push to Hugging Face Hub
                login()
                model.push_to_hub(model_key)
                file.write(model_key+ "\n")
                print(f"{model_key} added to saved_models.txt")

    # Load in model and quantize if not presaved
    if model_presaved == True: 
       # Load the pre-trained model if it's already saved
        model = AutoModelForCausalLM.from_pretrained("lesliehd/"+model_id)
    elif model_presaved == False:
        # Configuration for 4-bit quantization
        if bit_count == 4:
            nf4_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16
            )
            # Load the model with 4-bit quantization configuration
            model = AutoModelForCausalLM.from_pretrained(model_id, device_map='auto', quantization_config = nf4_config)
        # Configuration for 8-bit quantization
        if bit_count == 8:
            nf8_config = BitsAndBytesConfig(
                load_in_8bit=True
            )
            # Load the model with 8-bit quantization configuration
            model = AutoModelForCausalLM.from_pretrained(model_id, device_map='auto', quantization_config=nf8_config)
        if bit_count == 16:
            # Load the model with 16-bit quantization
            model = AutoModelForCausalLM.from_pretrained(model_id, device_map='auto', torch_dtype=torch.float16)
        if bit_count == 32:
            # Load the model without quantization (32-bit)
            model = AutoModelForCausalLM.from_pretrained(model_id, device_map='auto')
        print(f"Model Size: {model.get_memory_footprint():,} bytes")
    return model, device

def load_tokenizer(model_id):
    """
    Loads the tokenizer for the given model.

    Args:
        model_id (str): The identifier for the model.

    Returns:
        PreTrainedTokenizer: The loaded tokenizer.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    return tokenizer
            
def main():
    # Model identifier
    model_id = "mistralai/Mistral-7B-Instruct-v0.2"
    
    # Choose the quantization level
    bit_input = int(input("Enter the number of bits 4, 8, 16, or 32 (un-quantized): "))
    if bit_input not in [4, 8, 16, 32]:
        print("Invalid bit count. Loading model in int8 (8-bit) mode by default.")
        bit_input = 8

    # Load the model
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