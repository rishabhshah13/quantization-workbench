from scripts.mistral_quantize import load_model_quantized,load_tokenizer
from transformers import  AutoTokenizer

class Model:
    def __init__(self, model_name, bit_count,model_context):
        self.model, self.device = load_model_quantized(model_name, bit_count)
        self.tokenizer = load_tokenizer(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if model_context is None: 
            self.context = []
        else:
            self.context = model_context

    def get_output(self, user_input):
        self.context.append({"role": "user", "content": user_input})
        print(self.context)
        encodeds = self.tokenizer.apply_chat_template(self.context, return_tensors="pt")
        model_inputs = encodeds.to(self.device)
        generated_ids = self.model.generate(model_inputs, pad_token_id=self.tokenizer.pad_token_id, max_new_tokens = 1000, do_sample=True)
        decoded = self.tokenizer.batch_decode(generated_ids,clean_up_tokenization_spaces=True)
        self.context.append({"role": "assistant", "content": decoded[0]})
        return decoded[0]
