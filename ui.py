# ui.py

import gradio as gr
from model import Model  # Import the Model class
import traceback

models = {}

def compare_models(user_input, model_name, bit_counts):
    outputs = [None] * 4  # Initialize a list of 4 None values
    try:
        for i, bit_count in enumerate(bit_counts):
            key = f"{model_name}-{bit_count}-bit-quantized"
            if key not in models:
                models[key] = Model(model_name, int(bit_count))
            outputs[i] = models[key].get_output(user_input)
    except Exception as e:
        print(e)
        traceback.print_exc()
    return tuple(outputs)  # Convert the list to a tuple

inputs=[
    "text", 
    gr.Dropdown(["mistralai/Mistral-7B-Instruct-v0.2", "mistralai/Mistral-7B", "other_model"]), 
    gr.CheckboxGroup(["4", "8", "16", "32"]),
]

outputs = [gr.Textbox() for _ in range(4)]  # Adjust this number based on the maximum number of models you expect to compare

iface1 = gr.Interface(
    fn=compare_models, 
    inputs=inputs,
    outputs=outputs,
    title="Quantization Workbench",
    description="Compare various LLM models with each other and their quantized versions."
)

if __name__ == "__main__":
    iface1.launch(debug=True)
