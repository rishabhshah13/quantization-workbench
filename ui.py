import gradio as gr
from gradio import components
from scripts.mistral_quantize import load_model_quantized
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
import os
from huggingface_hub import login
import traceback


def get_model_output(user_input, model_name, bit_count):
    model = None
    device = None
    try:
        model, device = load_model_quantized(model_name, bit_count)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        messages = [{"role": "user", "content": user_input}]
        encodeds = tokenizer.apply_chat_template(messages, return_tensors="pt")
        model_inputs = encodeds.to(device)
        generated_ids = model.generate(model_inputs, pad_token_id=tokenizer.pad_token_id, max_new_tokens = 1000, do_sample=True)
        decoded = tokenizer.batch_decode(generated_ids)
        return decoded[0]
    except Exception as e:
        print(f"An error occurred while loading the model or generating output: {str(e)}")
        print(f"Model: {model}")
        print(f"Device: {device}")
        raise gr.Error(f"An error occurred: {str(e)}")


def compare_models(user_input, model_name, bit_counts):
    outputs = {}
    try:
        for bit_count in bit_counts:
            key = f"{model_name}-{bit_count}-bit-quantized"
            outputs[key] = get_model_output(user_input, model_name, bit_count)
    except Exception as e:
        print(e)
        traceback.print_exc()
    return outputs


inputs=[
    "text", 
    gr.Dropdown(["mistralai/Mistral-7B-Instruct-v0.2", "mistralai/Mistral-7B", "other_model"]), 
    gr.CheckboxGroup(["4", "8", "16", "32"])
]

outputs = gr.Textbox()

iface = gr.Interface(
    fn=compare_models, 
    inputs=inputs, 
    outputs = outputs,
    title="Quantization Workbench",
    description="Compare various LLM models with each other and their quantized versions."
)

if __name__ == "__main__":
    iface.launch(debug=True)