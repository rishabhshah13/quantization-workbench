from scripts.mistral_quantize import load_model_quantized, load_tokenizer as mistral_tokenizer  # Import functions for loading quantized model and tokenizer
from scripts.llama_awq import load_awq, load_tokenizer as llama_tokenizer # Import functions for loading quantized model and tokenizer

class Model:
    """Class to handle interaction with the language model."""
    
    def __init__(self, model_name, bit_count, model_context):
        """
        Initialize the Model instance.
        
        Parameters:
        - model_name (str): Name of the pre-trained language model.
        - bit_count (int): Number of bits for quantization.
        - model_context (list): Context information for the model.
        """
        # Load the quantized model and set device
        print(model_name)
        if 'mistral' in model_name:
            self.model, self.device = load_model_quantized(model_name, bit_count)
            self.tokenizer = mistral_tokenizer(model_name)
        elif 'Llama' in model_name:
            self.model, self.device = load_awq(model_name)
            self.tokenizer = llama_tokenizer(model_name)

        # Load the tokenizer

        # self.tokenizer = load_tokenizer(model_name)
        #self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Initialize model context
        if model_context is None: 
            self.context = []
        else:
            self.context = model_context

    def get_output(self, user_input):
        """
        Generate model output based on user input.
        
        Parameters:
        - user_input (str): User input text.
        
        Returns:
        - str: Generated text response from the model.
        """
        # Append user input to context
        self.context.append({"role": "user", "content": user_input})
        print(self.context)
        
        # Tokenize input and prepare model inputs
        encodeds = self.tokenizer.apply_chat_template(self.context, return_tensors="pt")
        model_inputs = encodeds.to(self.device)
        
        # Generate response from the model
        generated_ids = self.model.generate(model_inputs, pad_token_id=self.tokenizer.pad_token_id, max_new_tokens=1000, do_sample=True)
        decoded = self.tokenizer.batch_decode(generated_ids, clean_up_tokenization_spaces=True)
        
        # Append model output to context
        self.context.append({"role": "assistant", "content": decoded[0]})
        
        return decoded[0]  # Return the generated response
    
    def get_latest_response(self, conversation):
        """
        Extract the latest response from the conversation history.

        Parameters:
        - conversation (str): The conversation history.

        Returns:
        - str: The latest response in the conversation.
        """
        # Split the conversation by the [/INST] tag
        split_conversation = conversation.split("[/INST]")

        # The latest response is after the last [/INST] tag
        latest_response = split_conversation[-1]

        # Remove leading and trailing whitespace
        latest_response = latest_response.strip()

        return latest_response

