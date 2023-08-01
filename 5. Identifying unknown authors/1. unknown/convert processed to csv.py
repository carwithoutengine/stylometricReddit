import csv
import re


def preprocess_text(text):
    return text.strip()


def splitText(text, size):
    words = re.findall(r"\b\w+\b", text)
    return [words[i : i + size] for i in range(0, len(words), size)]


def makeCSV(input_file, output_file, reddit_user):
    with open(input_file, "r", encoding="utf-8") as file:
        text_data = file.read()

    # split the text into pieces of n words
    size = 174
    text_tukda = splitText(text_data, size)

    data_list = []
    for tukda in text_tukda:
        processed_tukda = preprocess_text(" ".join(tukda))
        if processed_tukda:  # for empty comments
            data_list.append({"text": processed_tukda, "author": reddit_user})

    with open(output_file, "w", newline="", encoding="utf-8") as file:
        fieldnames = ["text", "author"]
        writer = csv.DictWriter(file, fieldnames=fieldnames)

        writer.writeheader()
        writer.writerows(data_list)


if __name__ == "__main__":
    input_file = "c.txt"
    output_file = "../0. combined/AuthorC.csv"  # change the name of the csv file to whichever author you're testing
    reddit_user = "AuthorC"  # have to manually change username for each author

    makeCSV(input_file, output_file, reddit_user)