import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# Load the training CSV file (combined data)
input_file = "./0. combined/combined_data.csv"
df_train = pd.read_csv(input_file)

# further processing the training data, removing stopwords
stop_words = set(stopwords.words("english"))
df_train["processed_text"] = df_train["text"].apply(
    lambda x: " ".join([word for word in word_tokenize(x) if word not in stop_words])
)

# tokenizing
X_train = df_train["processed_text"]
y_train = df_train["author"]

# vectorizing data using TFIDF
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)

# training model
model = SVC(
    C=10,
    class_weight=None,
    degree=2,
    gamma=0.1,
    kernel="rbf",
)
model.fit(X_train_tfidf, y_train)

# Load the separate CSV file for test data with comments from different authors
test_input_file = (
    "./0. combined/unknown_data.csv"  # Replace with the path to your test CSV file
)
df_test = pd.read_csv(test_input_file)

# Preprocess the test data in the same way as the training data
df_test["processed_text"] = df_test["text"].apply(
    lambda x: " ".join([word for word in word_tokenize(x) if word not in stop_words])
)

# tokenizing
X_test = df_test["processed_text"]

# Vectorize the test data using the same TFIDF vectorizer from the training data
X_test_tfidf = vectorizer.transform(X_test)

# Testing on the test set
y_pred = model.predict(X_test_tfidf)

# Count the number of predictions for each author ('Author B' and 'Author C')
author_counts = pd.Series(y_pred).value_counts()

# Determine the author names based on the counts
author_b = author_counts.idxmax()
author_c = author_counts.idxmin()

print("Author B:", author_b)
print("Author C:", author_c)
