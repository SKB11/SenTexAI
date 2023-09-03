# src/pages/Text_Senti_Analyse.py

import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from twilio.rest import Client

nltk.download('punkt')
nltk.download('stopwords')

# Step 1: Load the dataset from the specified location
dataset_path = r'C:\Users\Sharddha K B\SenTexAI-main\Data\Dataset\s1.csv'
df = pd.read_csv(dataset_path)

# Step 2: Preprocess the data
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

# Step 3: Split the dataset into features and labels
X = df['text']
y = df['sentiment']

# Step 4: Convert text to numeric features using CountVectorizer
vectorizer = CountVectorizer()
X_vectorized = vectorizer.fit_transform(X)

# Step 5: Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42)

# Step 6: Build and train the Naive Bayes model
naive_bayes = MultinomialNB()
naive_bayes.fit(X_train, y_train)

# Step 7: Evaluate the model
y_pred = naive_bayes.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
confusion_mat = confusion_matrix(y_test, y_pred)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("Confusion Matrix:")
print(confusion_mat)

# Step 8: Prompt user for text input
text_input = input("Enter the text for sentiment analysis: ")

# Step 9: Preprocess the user input
cleaned_text_input = preprocess_text(text_input)

# Step 10: Vectorize the preprocessed text
text_vectorized = vectorizer.transform([cleaned_text_input])

# Step 11: Predict sentiment for the input text
predicted_sentiment = naive_bayes.predict(text_vectorized)

# Step 12: Print the result and professional messages
if predicted_sentiment[0] == 1:
    print("Positive sentiment")
    print("Have a Happy day! (: ")
else:
    print("Negative sentiment")
    print("Do not worry! Stay STRONG. Your help is on the way.")

    # Add code to send an SMS notification to the authority
    if __name__ == "__main__":
        # Replace these values with your Twilio credentials and phone numbers
        account_sid = 'give your twilio acct id'
        auth_token = 'give your twilio acct key'
        from_phone_number = '+give twilio generated mobile number'  # Your Twilio phone number
        authority_phone_number = '+give twilio verified mobile number'  # The authority's phone number

        # Initialize the Twilio client
        client = Client(account_sid, auth_token)

        # Customize the message
        sms_message = "Negative sentiment detected: " + text_input

        try:
            message = client.messages.create(
                body=sms_message,
                from_=from_phone_number,
                to=authority_phone_number
            )
            print(f"SMS sent to {authority_phone_number}: {message.sid}")
        except Exception as e:
            print(f"Failed to send SMS: {str(e)}")
