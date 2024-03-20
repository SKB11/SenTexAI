from flask import Flask, render_template, request
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

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
    return render_template('index1.html')

@app.route('/predict', methods=['POST'])
def predict():
    text_input = request.form['text_input']
    
    # Perform sentiment analysis
    predicted_sentiment = analyze_sentiment(text_input)

    if predicted_sentiment == 1:
        sentiment_label = "Positive sentiment"
        professional_message = "Have a Happy day! (: "
    else:
        sentiment_label = "Negative sentiment"
        professional_message = "Do not worry! Stay STRONG. Your help is on the way."

    return render_template('result1.html', text_input=text_input, sentiment_label=sentiment_label, professional_message=professional_message)

if __name__ == '__main__':
    app.run(debug=True)
