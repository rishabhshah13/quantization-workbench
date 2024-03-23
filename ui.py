import gradio as gr
from scripts.mistral_quantize import load_model_quantized

def get_model_output(user_input, model_name, bit_count):
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
        raise gr.Error(f"An error occurred: {str(e)}")

def compare_models(user_input, model_name, bit_counts):
    outputs = {}
    for bit_count in bit_counts:
        key = f"{model_name}-{bit_count}-bit-quantized"
        outputs[key] = get_model_output(user_input, model_name, bit_count)
    return outputs

iface = gr.Interface(
    fn=compare_models, 
    inputs=["text", gr.inputs.Dropdown(["mistralai/Mistral-7B-Instruct-v0.2", "mistralai/Mistral-7B", "other_model"]), gr.inputs.CheckboxGroup(["4", "8", "16", "32"])], 
    outputs="text",
    title="Quantization Workbench",
    description="Compare various LLM models with each other and their quantized versions."
)

if __name__ == "__main__":
    iface.launch(debug=True)
