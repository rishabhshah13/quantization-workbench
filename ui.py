# ui.py

import gradio as gr  # Import Gradio library for building UI
import traceback  # Import traceback module for error handling
from scripts.Model import Model # Import the Model class from model.py
from codecarbon import EmissionsTracker  # Import EmissionsTracker from codecarbon
import numpy as np

models = {}  # Dictionary to store instantiated models
context = {}  # Dictionary to store context information for each model
cumulative_emissions = {}  # Dictionary to store cumulative emissions for each model

def compare_models(user_input, model_name, bit_counts):
    """
    Compares models with different quantization levels and returns their outputs.

    Parameters:
    - user_input (str): The input text provided by the user.
    - model_name (str): The name of the model to compare.
    - bit_counts (list of str): List of quantization levels to compare.

    Returns:
    - list: A list containing the outputs of the models at each quantization level.
    """
    try:
        for i, bit_count in enumerate(bit_counts):
            key = f"{model_name}-{bit_count}-bit-quantized"

            if key in context.keys():
                model_context = context[key]
            else:
                model_context = None

            # Instantiate the EmissionsTracker
            tracker = EmissionsTracker()
            # Start the tracker
            tracker.start()

            if key not in models:
                models[key] = Model(model_name, int(bit_count), model_context)

            model_output = models[key].get_output(user_input)
            latest_response = models[key].get_latest_response(model_output)

            # Stop the tracker and get the emissions data
            emissions_data = tracker.stop()

            # Add the emissions data to the cumulative emissions
            if key in cumulative_emissions.keys():
                cumulative_emissions[key] += emissions_data
            else:
                cumulative_emissions[key] = emissions_data

            # Add the emissions data to the key
            key_with_emissions = f"{key} \n (Energy Consumption For This Inference: {np.round((emissions_data * 1000), 3)} Wh) \n (Cumulative Energy Consumption: {np.round((cumulative_emissions[key] * 1000), 3)} Wh)"

            if key in context.keys():
                outputs[i].append(['user', user_input])
                outputs[i].append([key_with_emissions, latest_response])
            else:
                outputs[i] = [['user', user_input], [key_with_emissions, latest_response]]

            context[key] = models[key].context
            del models[key]

    except Exception as e:
        print(e)
        traceback.print_exc()

    return outputs



# Define input components for the interface
inputs = [
    "text",  # Text input
    gr.Dropdown(["mistralai/Mistral-7B-Instruct-v0.2","TheBloke/Llama-2-7B-Chat-AWQ", "mistralai/Mistral-7B", "other_model"]),  # Dropdown for model selection
    gr.CheckboxGroup(["4", "8", "16", "32"]),  # Checkbox group for selecting quantization levels
]

# Define output components for the interface
outputs = [gr.Chatbot() for _ in range(4)]  # Chatbots to display model outputs

# Create Gradio interface with the defined inputs and outputs
iface1 = gr.Interface(
    fn=compare_models,  # Function to call when inputs are provided
    inputs=inputs,  # Input components
    outputs=outputs,  # Output components
    title="Quantization Workbench",  # Interface title
    description="Compare various LLM models with each other and their quantized versions - simultaneously chat with and compare the outputs and energy consumptions of different quantized models for the same prompts."  # Interface description
)

# Launch the interface if this script is executed directly
if __name__ == "__main__":
    iface1.launch(debug=True)  # Launch the interface in debug mode
