import random
import math

def build_markov_chain(text):
    """
    Builds a Markov chain model from the given text.

    Args:
        text (str): The input text used to build the Markov chain model.

    Returns:
        dict: The Markov chain model.
    """
    words = text.split()
    model = {}
    
    # Build the Markov chain model
    for i in range(len(words)-1):
        word, next_word = words[i], words[i+1]
        if word not in model:
            model[word] = {}
        if next_word not in model[word]:
            model[word][next_word] = 0
        model[word][next_word] += 1
    
    # Convert counts to probabilities
    for word, following in model.items():
        total_following = sum(following.values())
        for next_word in following:
            model[word][next_word] /= total_following
    
    return model

def generate_text(model, start_word, length=50):
    """
    Generates text using the provided Markov chain model.

    Args:
        model (dict): The Markov chain model.
        start_word (str): The starting word for text generation.
        length (int, optional): The length of the generated text. Defaults to 50.

    Returns:
        str: The generated text.
    """
    current_word = start_word
    text = [current_word]
    
    # Generate text based on the Markov chain model
    for _ in range(length-1):
        if current_word not in model or not model[current_word]:
            break  # Stop if the current word is not in the model
        next_words = list(model[current_word].keys())
        probabilities = list(model[current_word].values())
        current_word = random.choices(next_words, weights=probabilities, k=1)[0]
        text.append(current_word)
    
    return ' '.join(text)

def calculate_perplexity(model, test_text):
    """
    Calculates the perplexity of the provided Markov chain model on the test text.

    Args:
        model (dict): The Markov chain model.
        test_text (str): The test text for which perplexity is calculated.

    Returns:
        float: The perplexity of the model on the test text.
    """
    test_words = test_text.split()
    log_prob = 0
    count = 0
    
    # Calculate the perplexity
    for i in range(len(test_words)-1):
        word, next_word = test_words[i], test_words[i+1]
        if word in model and next_word in model[word]:
            probability = model[word][next_word]
            log_prob += math.log(probability)
            count += 1
    
    perplexity = math.exp(-log_prob / count) if count > 0 else float('inf')
    return perplexity

def main():
    ## Parameters needed: input_text, test_text, start_word

    with open('./wikipedia_data.txt', 'r', encoding='utf-8') as file:
        input_text = file.read()

    test_text = "since she was a kid"
    start_word = "since"

    # Build the Markov chain model
    model = build_markov_chain(input_text)
    
    # Generate text using the model
    generated_text = generate_text(model, start_word, 100)
    print("Generated Text:")
    print(generated_text)
    
    # Calculate and print the perplexity of the model on the test text
    perplexity = calculate_perplexity(model, test_text)
    print("\nModel Perplexity on Test Text:", perplexity)

if __name__ == "__main__":
    main()
