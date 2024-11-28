from flask import Flask, request, jsonify
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import re
from flask_cors import CORS
# Load pre-trained model and other necessary components
model = joblib.load('senti_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')  # Assuming you saved your vectorizer
label_encoder = joblib.load('label_encoder.pkl')  # Assuming you saved your label encoder

app = Flask(__name__)

CORS(app)


# Function for preprocessing the text (same as used during model training)
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove non-alphabetical characters
    return text

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()  # Receive the data as JSON
    review = data['review']  # The review text to be analyzed

    # Preprocess and vectorize the review
    cleaned_review = preprocess_text(review)
    vectorized_review = vectorizer.transform([cleaned_review]).toarray()

    # Predict sentiment
    sentiment = model.predict(vectorized_review)
    sentiment_label = label_encoder.inverse_transform(sentiment)[0]

    # Return the prediction as a JSON response
    return jsonify({'sentiment': sentiment_label})


if __name__ == '__main__':
    app.run(debug=True)
