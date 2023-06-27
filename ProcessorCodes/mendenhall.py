import re
import nltk
import matplotlib.pyplot as plt
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from nltk.corpus import stopwords

def preprocess_text(text):
    # Remove emojis
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002500-\U00002BEF"  # Chinese characters
                               u"\U00002702-\U000027B0"
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               u"\U0001f926-\U0001f937"
                               u"\U00010000-\U0010ffff"
                               u"\u2640-\u2642"
                               u"\u2600-\u2B55"
                               u"\u200d"
                               u"\u23cf"
                               u"\u23e9"
                               u"\u231a"
                               u"\ufe0f"  # dingbats
                               u"\u3030"
                               "]+", flags=re.UNICODE)
    text = emoji_pattern.sub('', text)

    # Remove URLs
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    text = url_pattern.sub('', text)

    # Remove special characters and convert to lowercase
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = text.lower()

    return text

def tokenize_text(text):
    tokens = word_tokenize(text)
    return tokens

def mendenhall_method(tokens):
    fdist = FreqDist(tokens)
    mendenhall_list = fdist.most_common(30)
    return mendenhall_list

def plot_density_plots(dataset, division_size):
    num_words = len(dataset)
    num_divisions = num_words // division_size

    fig, ax = plt.subplots(figsize=(10, 6))

    for i in range(num_divisions):
        start_index = i * division_size
        end_index = start_index + division_size

        division = dataset[start_index:end_index]

        mendenhall_result = mendenhall_method(division)

        words, frequencies = zip(*mendenhall_result)
        line_label = f'Division {i+1}'

        ax.plot(words, frequencies, label=line_label)

    ax.set_xlabel('Words')
    ax.set_ylabel('Frequency')
    ax.legend()
    ax.grid(True)

    plt.show()

# Example usage
input_file_path = 'C:/IITB/Year 1/Non-Academic/SoC/reddit_catbreadmash_comments-43k.txt'

# Read the input file
with open(input_file_path, 'r', encoding='utf-8') as file:
    text = file.read()

# Preprocess the text
preprocessed_text = preprocess_text(text)

# Tokenize the preprocessed text
tokenized_text = tokenize_text(preprocessed_text)

# Set the desired division size
division_size = 1500

# Apply Mendenhall's method and plot the density plots
plot_density_plots(tokenized_text, division_size)