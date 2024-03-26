import wikipedia
import re

def get_random_wikipedia_article():
    """
    Fetches a random Wikipedia article and returns its cleaned text.

    Returns:
        str: The cleaned text of the random Wikipedia article.
    """
    # Set the language to English
    wikipedia.set_lang("en")
    
    # Fetch a random page
    random_page = wikipedia.page(wikipedia.random(pages=1))
    text = random_page.content  # Get the full text of the article
    
    # Attempt to remove section titles using a regular expression
    cleaned_text = re.sub(r'==.*?==+', '', text)
    # Further clean up by removing extra newlines and leading/trailing whitespace
    cleaned_text = re.sub(r'\n\s*\n', '\n\n', cleaned_text).strip()
    
    return cleaned_text

def collect_wikipedia_data_and_save_to_file(number_of_articles=10, file_name='wikipedia_data.txt'):
    """
    Collects Wikipedia article data and saves it to a file.

    Args:
        number_of_articles (int, optional): The number of articles to collect. Defaults to 10.
        file_name (str, optional): The name of the file to save the data to. Defaults to 'wikipedia_data.txt'.
    """
    with open(file_name, 'w', encoding='utf-8') as file:
        # Loop to collect the specified number of articles
        for _ in range(number_of_articles):
            try:
                # Get text from a random Wikipedia article
                text = get_random_wikipedia_article()
                # Write the text to the file
                file.write(text + '\n\n') 
            except Exception as e:
                print(f"An error occurred: {e}")

# Parameters
NUMBER_OF_ARTICLES = 150
FILE_NAME = 'wikipedia_data.txt'  

# Collect Wikipedia data and save to file
collect_wikipedia_data_and_save_to_file(NUMBER_OF_ARTICLES, FILE_NAME)

# Print a confirmation message
print(f'Data from {NUMBER_OF_ARTICLES} articles has been saved to "{FILE_NAME}".')
