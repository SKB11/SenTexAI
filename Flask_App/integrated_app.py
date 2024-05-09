# All the modules work on 3.12.0 version of python

from flask import Flask, render_template, request
import pandas as pd
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, precision_score, f1_score, recall_score
import re
from nltk.stem import PorterStemmer
from twilio.rest import Client
import sys

# Define preprocess_text function
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = nltk.word_tokenize(text)
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(word) for word in tokens]
    preprocessed_text = ' '.join(tokens)
    return preprocessed_text

# Read data - Give data path
df = pd.read_csv('/Users/bharath/Desktop/REVA/SenTexAI/Data/Dataset/newData.txt', sep='\t', names=['liked', 'text'])

# Preprocess the text data
df['text'] = df['text'].apply(preprocess_text)

# TF-IDF Vectorizer
vectorizer = TfidfVectorizer(stop_words='english')

y = df.liked
X = vectorizer.fit_transform(df.text)

# Splitting the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Naive Bayes classifier
clf_nb = MultinomialNB()
clf_nb.fit(X_train, y_train)

# Read keywords from file
with open('/Users/bharath/Desktop/REVA/SenTexAI/Data/Dataset/keywords.txt', 'r') as file:
    keywords = [word.strip() for line in file for word in line.split(',')]

# Initialize Flask app
app = Flask(__name__)

# Function for sending SMS notification
def send_sms_notification(message):
    # Replace these values with your Twilio credentials and phone numbers
    account_sid = 'ACa13f657752d57143236caf3d995f5af0' 
    auth_token = '60e94a1465d3ffb0fef50f07e7dbbef7'
    from_phone_number = '+12075696420'  # Your Twilio phone number
    authority_phone_number = '+918431565515'  # The authority's phone number

    # Initialize the Twilio client
    client = Client(account_sid, auth_token)

    try:
        message = client.messages.create(
            body=message,
            from_=from_phone_number,
            to=authority_phone_number
        )
        print(f"SMS sent to {authority_phone_number}: {message.sid}")
    except Exception as e:
        print(f"Failed to send SMS: {str(e)}")

# Function for sentiment analysis
def analyze_sentiment(input_text):
    # Check for specific phrases
    if 'ambulance' in input_text.lower() or 'labour' in input_text.lower():
        emergency_message = "An emergency message has been initiated to contact an ambulance."
        send_sms_notification(emergency_message)
        return None, emergency_message
    elif 'fire' in input_text.lower():
        emergency_message = "An emergency message has been initiated to contact a fire truck."
        send_sms_notification(emergency_message)
        return None, emergency_message
    elif 'thief' in input_text.lower() or 'robber' in input_text.lower() or 'robbing' in input_text.lower():
        emergency_message = "An emergency message has been initiated to contact the police."
        send_sms_notification(emergency_message)
        return None, emergency_message
    
    # Rule-based classification
    for keyword in keywords:
        if keyword in input_text:
            send_sms_notification("Negative sentiment detected: " + input_text)
            return 0, None  # Negative sentiment
    
    # Predict sentiment using Naive Bayes
    sentiment_nb = clf_nb.predict(vectorizer.transform([input_text]))
    if sentiment_nb[0] == 0:  # Negative sentiment
        send_sms_notification("Negative sentiment detected: " + input_text)
    return sentiment_nb[0], None  # Return the predicted sentiment (1 for positive, 0 for negative)


# Route for homepage
@app.route('/')
def index():
    converted_text = request.args.get('converted_text', '')  # Get the converted text from query parameter
    return render_template('integrated_index.html', converted_text=converted_text)

# Route for sentiment prediction
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        text_input = request.form['text_input']  # Get the text input from the form

        # Perform sentiment analysis
        predicted_sentiment, emergency_message = analyze_sentiment(text_input)

        if predicted_sentiment == 1:
            sentiment_label = "Positive sentiment"
            professional_message = "Have a Happy day! (: "
        else:
            sentiment_label = "Negative sentiment"
            professional_message = "Do not worry! Stay STRONG. Your help is on the way."

        return render_template('integrated_result.html', text_input=text_input, sentiment_label=sentiment_label, professional_message=professional_message, emergency_message=emergency_message)
    
    send_sms_notification("Clicks Detected: Sending emergency message.")
    return render_template('integrated_result.html', text_input="Clicks Detected (NO TEXT INPUT) ", sentiment_label="Dis-Stress Signal Sent", professional_message="STAY CALM, HELP IS ON THE WAY!")

if __name__ == '__main__':
    app.run(debug=True)
