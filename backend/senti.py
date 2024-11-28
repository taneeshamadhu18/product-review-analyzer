import pandas as pd
import numpy as np
import re
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# Load the dataset (replace with the actual path or URL to your dataset)
data = pd.read_csv("IMDB Dataset.csv")

# Stop words list (you can expand this list)
stopwords = set([
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", 
    "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", 
    "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", 
    "theirs", "themselves", "what", "which", "who", "whom", "this", "that", 
    "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", 
    "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the", 
    "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", 
    "for", "with", "about", "against", "between", "into", "through", "during", 
    "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", 
    "on", "off", "over", "under", "again", "further", "then", "once", "here", 
    "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", 
    "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", 
    "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", 
    "should", "now", "d", "ll", "m", "o", "re", "ve", "y", "ain", "aren", "couldn", 
    "didn", "doesn", "hadn", "hasn", "haven", "isn", "ma", "mightn", "mustn", "needn", 
    "shan", "shouldn", "wasn", "weren", "won", "wouldn"
])

# Preprocessing function: Remove punctuation, tokenize and remove stop words
def preprocess_text(text):
    # Convert text to lowercase
    text = text.lower()
    
    # Remove HTML tags (e.g., <br> and other HTML-like artifacts)
    text = re.sub(r'<.*?>', '', text)
    
    # Remove non-alphanumeric characters (keeping spaces)
    text = re.sub(r'[^a-z0-9\s]', '', text)
    
    # Remove multiple spaces and strip leading/trailing spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Tokenize the text and remove stopwords
    tokens = text.split()
    tokens = [word for word in tokens if word not in stopwords]
    
    return " ".join(tokens)


# Apply the preprocessing function to the dataset
data['cleaned_text'] = data['review'].apply(preprocess_text)

# Convert sentiments (e.g., "positive" and "negative") to numeric values using LabelEncoder
label_encoder = LabelEncoder()
data['sentiment'] = label_encoder.fit_transform(data['sentiment'])

# Extract features using TF-IDF vectorization
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(data['cleaned_text']).toarray()

# Target variable: sentiment (0 for negative, 1 for positive)
y = data['sentiment']

# Split data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize Logistic Regression model and train it
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Evaluate the model accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

# Streamlit interface
st.title('Sentiment Analysis with IMDB Reviews')

# User input for review text
input_text = st.text_area("Enter a Movie Review")

# Predict sentiment when user presses 'Predict'
if st.button('Predict Sentiment'):
    if input_text:
        # Preprocess and vectorize the input text
        cleaned_text = preprocess_text(input_text)
        vectorized_text = vectorizer.transform([cleaned_text]).toarray()
        
        # Predict sentiment: 0 for Negative, 1 for Positive
        sentiment = model.predict(vectorized_text)
        
        # Output the result
        st.write(f"Predicted Sentiment: {label_encoder.inverse_transform(sentiment)[0]}")
    else:
        st.write("Please enter a movie review.")
import joblib

# Save the trained model
joblib.dump(model, 'senti_model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')
joblib.dump(label_encoder, 'label_encoder.pkl')
