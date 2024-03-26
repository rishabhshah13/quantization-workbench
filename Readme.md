# Quantization Workbench
This project provides a tool for benchmarking and comparing different quantized versions of models in terms of their output and energy consumption.

## Introduction to Quantization
Quantization is a process that reduces the number of bits that represent a number. In the context of deep learning, quantization is a technique used to perform computation and storage in lower precision. This results in smaller model size and faster computation, often with minimal impact on the model’s accuracy.

## Overview of the Tool
The tool consists of several Python scripts that work together to load a model, quantize it to a specified bit count, and then generate responses from the model. The tool also measures the size of the model in memory, which can be used as a proxy for energy consumption.

## Key Components
1. mistral_quantize.py: This script contains the load_model_quantized function, which loads a pre-trained model and quantizes it to a specified bit count. The function uses the BitsAndBytesConfig class from the transformers library to configure the quantization settings.

2. Model.py: This script defines a Model class that handles interaction with the language model. The class has methods to generate model output based on user input (get_output) and to extract the latest response from the conversation history (get_latest_response).

3. main function: The main function in the mistral_quantize.py script prompts the user to enter the number of bits for quantization, loads the model and tokenizer, and then enters a loop where it continually prompts the user for input, generates a response from the model, and prints the response.

4. UI: ui.py sets up a UI using gradio to simultaneously prompt and compare the outputs and energy consumptions of different quantized models simultaneously in a web chatbot interface for each separate selected quantized model.

## Installation Instructions

Follow the steps below to setup the repo based on your environment:

### Conda Installation Instructions

```bash
conda create --prefix "C:\\Users\\rs659\\Desktop\\quantization-workbench\\wincondaprojenv" python=3.9
conda activate "C:\\Users\\rs659\\Desktop\\quantization-workbench\\wincondaprojenv"
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121
pip install transformers==4.37.0 bitsandbytes==0.43.0 accelerate==0.27.2 ipywidgets
onda install cudatoolkit
conda install cudnn

!git clone https://github.com/casper-hansen/AutoAWQ_kernels 
cd AutoAWQ_kernels 
!pip install -e . 
```

If you are unable to clone the repo on Windows due to Filename too long error, run the following:
```git config --global core.longpaths true```

### MacOS Installation Instructions
```bash
conda create --prefix "/Users/rishabhshah/Desktop/quantization-workbench/wincondaprojenv" python=3.9
conda activate "/Users/rishabhshah/Desktop/quantization-workbench/wincondaprojenv"
pip3 install torch torchvision torchaudio   
pip install transformers==4.37.0 bitsandbytes==0.43.0 accelerate==0.27.2 ipywidgets
```


### WSL Installation Instructions
```bash
cd /mnt/c/Users/rs659/Desktop/quantization-workbench
export PATH=~/anaconda3/bin:$PATH
conda create --prefix=/mnt/c/Users/rs659/Desktop/quantization-workbench/wslenv python=3.9
conda activate /mnt/c/Users/rs659/Desktop/quantization-workbench/wslenv
conda install -p /mnt/c/Users/rs659/Desktop/quantization-workbench/wslenv ipykernel --update-deps --force-reinstall
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121
pip install git+https://github.com/casper-hansen/AutoAWQ.git
pip install vllm
pip install -U ipywidgets
```

## Run UI and Compare Models

To launch the UI and start chattting with the different models simultaneously, simply run the UI script:

```python -m ui```

## How the Tool Works

The tool works by first loading a pre-trained model using the load_model_quantized function. The user specifies the number of bits for quantization (4, 8, 16, or 32). If the model has been previously quantized and saved, it is loaded directly. Otherwise, the model is quantized according to the specified bit count using the BitsAndBytesConfig class, and then saved for future use.

The Model class is used to handle interaction with the language model. The get_output method appends the user input to the model context, tokenizes the input, generates a response from the model, and then appends the model output to the context. The get_latest_response method is used to extract the latest response from the conversation history.

The main function prompts the user to enter the number of bits for quantization, loads the model and tokenizer, and then enters a loop where it continually prompts the user for input, generates a response from the model, and prints the response. The size of the model in memory is printed after the model is loaded, which can be used as a measure of energy consumption.

## Example Usage

```python
from scripts.mistral_quantize import load_model_quantized, load_tokenizer

model_id = "mistralai/Mistral-7B-Instruct-v0.2"
bit_input = 8  # Specify the number of bits for quantization

# Load the quantized model and tokenizer
model, device = load_model_quantized(model_id, bit_input)
tokenizer = load_tokenizer(model_id)

# Create an instance of the Model class
model_instance = Model(model, tokenizer)

# Generate a response from the model
user_input = "Tell me about AI"
response = model_instance.get_output(user_input)
print("Model response:", response)
```

This will load the specified model, quantize it to 8 bits, and then generate a response to the input “Tell me about AI”. The response is then printed to the console. The size of the model in memory is also printed after the model is loaded.

## Conclusion
This tool provides a simple and effective way to benchmark and compare different quantized versions of models. By measuring the size of the model in memory and the quality of the model’s output, we can gain insights into the trade-offs between model size, computation speed, and model performance. This can be particularly useful in resource-constrained environments where model size and computation speed are critical. The tool provides a way to compare energy consumption of these models in any context/environment in which they are run.

## References
Hugging Face Transformers
Quantization in PyTorch