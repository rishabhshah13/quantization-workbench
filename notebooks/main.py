import torch
from mModel import mModel


def load_model(model_name,bit_input):

    model_obj = mModel(model_name,bit_input)

    model_obj.load_bnb()

    return model_obj



def main():
    
    model_name = int(input("Which model would you like to use? 1. Mistral 2. Llama:  "))
    if model_name not in [1, 2]:
        print("Invalid model selected. Loading mistral mode by default.")
        model_name = 'mistral'

    if model_name == 1:
        model_name = 'mistral'
    if model_name == 2:
        model_name = 'llama2'


    inference_mode = int(input("1. Single model or 2. Multi model Inference mode    "))
    if inference_mode not in [1,2]:
        print("Invalid bit count. Loading model in int8 (8-bit) mode by default.")
        inference_mode = 1
    #     model_obj = load_model(model_name,bit_input)


    if inference_mode == 1:
        bit_input = int(input("Enter the number of bits 4, 8, 16, or 32 (un-quantized): "))
        if bit_input not in [4, 8, 16, 32]:
            print("Invalid bit count. Loading model in int8 (8-bit) mode by default.")
            bit_input = 8
        model_obj = load_model(model_name,bit_input)
        model_obj.start_inference()

    else:
        text = str(input("Text Input for the models \t"))    
        for bit_input in [4,8,16,32]:
            model_obj = load_model(model_name,bit_input)
            model_obj.single_inference(text)
        
        # Don't use
    

if __name__ == "__main__":
    main()