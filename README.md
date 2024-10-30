Project Overview
This project involves building a web-based toxicity detection system to classify text comments as toxic or non-toxic. Using natural language processing (NLP) and machine learning, it automates moderation for platforms hosting user-generated content.

Files
app.py: Flask-based web application to classify comments.
toxicity_model.pkl: Pre-trained machine learning model for toxicity detection.
tf_idf_vectorizer.pkl: Vectorizer for transforming text input for the model.
index.html: User interface for comment input and toxicity output.
Setup
Install dependencies: pip install Flask scikit-learn nltk.
Run app.py to start the server.
Access the app on http://localhost:5000.
Usage
Enter a comment, and the app will classify it as toxic or non-toxic.
