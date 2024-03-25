import wikipedia
import re

def get_random_wikipedia_article():
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
    with open(file_name, 'w', encoding='utf-8') as file:
        for _ in range(number_of_articles):
            try:
                text = get_random_wikipedia_article()
                file.write(text + '\n\n') 
            except Exception as e:
                print(f"An error occurred: {e}")

NUMBER_OF_ARTICLES = 150
FILE_NAME = 'wikipedia_data.txt'  
collect_wikipedia_data_and_save_to_file(NUMBER_OF_ARTICLES, FILE_NAME)

print(f'Data from {NUMBER_OF_ARTICLES} articles has been saved to "{FILE_NAME}".')
