import torch
from mModel import mModel

def load_model(model_name,bit_input):

    model = mModel(model_name,bit_input)
    model = model.load()

    return model



def main():
    
    model_name = int(input("Which model would you like to use? 1. Mistral 2. Llama?"))
    if model_name not in [1, 2]:
        print("Invalid model selected. Loading mistral mode by default.")
        model_name = 'mistral'

    bit_input = int(input("Enter the number of bits 4, 8, 16, or 32 (un-quantized): "))
    if bit_input not in [4, 8, 16, 32]:
        print("Invalid bit count. Loading model in int8 (8-bit) mode by default.")
        bit_input = 8

    inference_mode = int(input("1. Single model inference mode or 2. Multi model"))
    if inference_mode not in [1,2]:
        print("Invalid bit count. Loading model in int8 (8-bit) mode by default.")
        inference_mode = 1
        model, device = load_model(model_name,bit_input)

    if inference_mode == 1:
        model = load_model(model_name,bit_input)
        model.start_inference()
    else:
        model = load_model(model_name,bit_input)
        # Don't use
    

if __name__ == "__main__":
    main()