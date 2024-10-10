# app.py

import pickle
import pandas as pd
from flask import Flask, render_template, request
from nltk import WordNetLemmatizer, pos_tag, word_tokenize
from nltk.corpus import wordnet
import re

# Load the pre-trained model and TF-IDF vectorizer
model_path = "models/toxicity_model.pkt"
tf_idf_path = "models/tf_idf.pkt"

with open(model_path, "rb") as model_file:
    model = pickle.load(model_file)

with open(tf_idf_path, "rb") as vectorizer_file:
    tf_idf_vectorizer = pickle.load(vectorizer_file)

# Initialize Flask app
app = Flask(__name__)

# Helper function to preprocess text
def prepare_text(text):
    wordnet_lemmatizer = WordNetLemmatizer()

    def get_wordnet_pos(treebank_tag):
        if treebank_tag.startswith('J'):
            return wordnet.ADJ
        elif treebank_tag.startswith('V'):
            return wordnet.VERB
        elif treebank_tag.startswith('N'):
            return wordnet.NOUN
        elif treebank_tag.startswith('R'):
            return wordnet.ADV
        else:
            return wordnet.NOUN

    text = re.sub(r'[^a-zA-Z\']', ' ', text)
    text = text.split()
    text = ' '.join(text)
    text = word_tokenize(text)
    text = pos_tag(text)

    lemma = [wordnet_lemmatizer.lemmatize(i[0], pos=get_wordnet_pos(i[1])) for i in text]
    return ' '.join(lemma)

@app.route('/', methods=['GET', 'POST'])
def index():
    result = ""
    if request.method == 'POST':
        user_input = request.form['comment']
        processed_input = prepare_text(user_input)
        tf_idf_input = tf_idf_vectorizer.transform([processed_input])
        prediction = model.predict(tf_idf_input)

        if prediction[0] == 1:  # Assuming 1 is toxic
            result = "Toxic Comment"
        else:
            result = "Non-Toxic Comment"

    return render_template('index.html', result=result)

# Vercel requires the app to be callable as a module
# Expose the app to Vercel
if __name__ == '__main__':
    app.run(debug=True)

