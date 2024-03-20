from flask import Flask, render_template, request, redirect, url_for
import os
import speech_recognition as sr
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

app = Flask(__name__)

UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static/uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

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

@app.route('/convert', methods=['POST'])
def convert():
    r = sr.Recognizer()

    if 'audio_file' not in request.files:
        return "No file part"

    audio_file = request.files['audio_file']

    if audio_file.filename == '':
        return "No selected file"

    audio_file_path = os.path.join(app.config['UPLOAD_FOLDER'], audio_file.filename)
    audio_file.save(audio_file_path)

    with sr.AudioFile(audio_file_path) as source:
        audio = r.listen(source)
        try:
            text = r.recognize_google(audio)
            return redirect(url_for('predict', text_input=text))  # Redirect to predict route with the converted text as query parameter
        except sr.UnknownValueError:
            return "Sorry, could not understand audio"
        except sr.RequestError as e:
            return f"Could not request results; {e}"

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    text_input = request.args.get('text_input', '')  # Get the text input from query parameter or form
    
    if request.method == 'POST':
        text_input = request.form['text_input']
    
    # Perform sentiment analysis
    predicted_sentiment = analyze_sentiment(text_input)

    if predicted_sentiment == 1:
        sentiment_label = "Positive sentiment"
        professional_message = "Have a Happy day! (: "
    else:
        sentiment_label = "Negative sentiment"
        professional_message = "Do not worry! Stay STRONG. Your help is on the way."

    return render_template('integrated_result.html', text_input=text_input, sentiment_label=sentiment_label, professional_message=professional_message)

if __name__ == '__main__':
    app.run(debug=True)
