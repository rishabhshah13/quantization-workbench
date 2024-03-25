# ui.py

import gradio as gr  # Import Gradio library for building UI
import traceback  # Import traceback module for error handling
from scripts.Model import Model # Import the Model class from model.py

models = {}  # Dictionary to store instantiated models
context = {}  # Dictionary to store context information for each model


def compare_models(user_input, model_name, bit_counts):
    """
    Compares models with different quantization levels and returns their outputs.

    Parameters:
    - user_input (str): The input text provided by the user.
    - model_name (str): The name of the model to compare.
    - bit_counts (list of str): List of quantization levels to compare.

    Returns:
    - tuple: A tuple containing the outputs of the models at each quantization level.
    """
    outputs = [None] * 4  # Initialize a list of 4 None values to store model outputs
    try:
        for i, bit_count in enumerate(bit_counts):
            # Generate a unique key for each model based on model name and quantization level
            key = f"{model_name}-{bit_count}-bit-quantized"

            # Check if context for this model already exists
            if key in context.keys():
                model_context = context[key]
            else:
                model_context = None

            # Check if the model instance already exists, if not create a new one
            if key not in models:
                models[key] = Model(model_name, int(bit_count), model_context)
            
            # Get output from the model and store it in outputs list
            outputs[i] = models[key].get_output(user_input)

            # Update context for the model
            context[key] = models[key].context

            # Delete the model instance after use to free up memory
            del models[key]

    except Exception as e:
        # Print and log any exceptions that occur during model comparison
        print(e)
        traceback.print_exc()

    return tuple(outputs)  # Convert the list to a tuple and return


# Define input components for the interface
inputs = [
    "text",  # Text input
    gr.Dropdown(["mistralai/Mistral-7B-Instruct-v0.2", "mistralai/Mistral-7B", "other_model"]),  # Dropdown for model selection
    gr.CheckboxGroup(["4", "8", "16", "32"]),  # Checkbox group for selecting quantization levels
]

# Define output components for the interface
outputs = [gr.Textbox() for _ in range(4)]  # Textboxes to display model outputs

# Create Gradio interface with the defined inputs and outputs
iface1 = gr.Interface(
    fn=compare_models,  # Function to call when inputs are provided
    inputs=inputs,  # Input components
    outputs=outputs,  # Output components
    title="Quantization Workbench",  # Interface title
    description="Compare various LLM models with each other and their quantized versions."  # Interface description
)

# Launch the interface if this script is executed directly
if __name__ == "__main__":
    iface1.launch(debug=True)  # Launch the interface in debug mode
