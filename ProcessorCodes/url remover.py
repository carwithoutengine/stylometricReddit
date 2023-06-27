import re

def remove_emojis(text):
    # Emoji pattern to match emojis
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
    # Remove emojis from the text
    text_without_emojis = emoji_pattern.sub('', text)
    return text_without_emojis

def remove_urls(text):
    # URL pattern to match URLs
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    # Remove URLs from the text
    text_without_urls = url_pattern.sub('', text)
    return text_without_urls

# Read the input text file
input_file_path = 'path/to/input/file.txt'
with open(input_file_path, 'r', encoding='utf-8') as file:
    text = file.read()

# Remove emojis from the text
text_without_emojis = remove_emojis(text)

# Remove URLs from the text
text_without_emojis_or_urls = remove_urls(text_without_emojis)

# Write the text without emojis or URLs to a new file
output_file_path = 'path/to/output/file.txt'
with open(output_file_path, 'w', encoding='utf-8') as file:
    file.write(text_without_emojis_or_urls)