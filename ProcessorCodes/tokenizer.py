import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize

def tokenize_text_file(input_file_path):
    with open(input_file_path, 'r', encoding='utf-8') as file:
        text = file.read()

    # Tokenization
    tokens = word_tokenize(text)

    return tokens

# Example usage
input_file_path = 'C:/IITB/Year 1/Non-Academic/SoC/pre/rcatbreadmashprocesssed.txt'
tokenized_text = tokenize_text_file(input_file_path)

print(tokenized_text)