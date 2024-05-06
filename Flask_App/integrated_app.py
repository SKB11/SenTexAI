from flask import Flask, render_template, request
import speech_recognition as sr
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Load the dataset
dataset_path = r'C:\Users\Sharddha K B\SenTexAI-main\Data\Dataset\s1.csv'
df = pd.read_csv(dataset_path)

# Preprocess the data
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = nltk.word_tokenize(text)
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(word) for word in tokens]
    preprocessed_text = ' '.join(tokens)
    return preprocessed_text

df['text'] = df['text'].apply(preprocess_text)

# Split the dataset into features and labels
X = df['text']
y = df['sentiment']

# Convert text to numeric features using CountVectorizer
vectorizer = CountVectorizer()
X_vectorized = vectorizer.fit_transform(X)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42)

# Build and train the Naive Bayes model
naive_bayes = MultinomialNB()
naive_bayes.fit(X_train, y_train)

# Evaluate the model
y_pred = naive_bayes.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
confusion_mat = confusion_matrix(y_test, y_pred)

print("Model Evaluation Metrics:")
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("Confusion Matrix:")
print(confusion_mat)

app = Flask(__name__)

# Load NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Load the dataset
dataset_path = r'C:\Users\Sharddha K B\SenTexAI-main\Data\Dataset\s1.csv'
df = pd.read_csv(dataset_path)

# Preprocess the data
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = nltk.word_tokenize(text)
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(word) for word in tokens]
    preprocessed_text = ' '.join(tokens)
    return preprocessed_text

df['text'] = df['text'].apply(preprocess_text)

# Split the dataset into features and labels
X = df['text']
y = df['sentiment']

# Convert text to numeric features using CountVectorizer
vectorizer = CountVectorizer()
X_vectorized = vectorizer.fit_transform(X)

# Build and train the Naive Bayes model
naive_bayes = MultinomialNB()
naive_bayes.fit(X_vectorized, y)

# Function for sentiment analysis
def analyze_sentiment(input_text):
    cleaned_text_input = preprocess_text(input_text)
    text_vectorized = vectorizer.transform([cleaned_text_input])
    predicted_sentiment = naive_bayes.predict(text_vectorized)
    return predicted_sentiment[0]  # Return the predicted sentiment (1 for positive, 0 for negative)

@app.route('/')
def index():
    converted_text = request.args.get('converted_text', '')  # Get the converted text from query parameter
    return render_template('integrated_index.html', converted_text=converted_text)

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        text_input = request.form['text_input']  # Get the text input from the form

        # Perform sentiment analysis
        predicted_sentiment = analyze_sentiment(text_input)

        if predicted_sentiment == 1:
            sentiment_label = "Positive sentiment"
            professional_message = "Have a Happy day! (: "
        else:
            sentiment_label = "Negative sentiment"
            professional_message = "Do not worry! Stay STRONG. Your help is on the way."

        return render_template('integrated_result.html', text_input=text_input, sentiment_label=sentiment_label, professional_message=professional_message)

    return render_template('integrated_predict.html')

if __name__ == '__main__':
    app.run(debug=True)
