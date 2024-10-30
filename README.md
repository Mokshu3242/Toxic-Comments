## Toxic Comments Classification (NLP Project)

### Overview
The Toxic Comments Classification project is a Flask-based web application that uses Natural Language Processing (NLP) and machine learning to classify text as toxic or non-toxic. It automates the detection of harmful content in real-time, aimed at supporting online moderation efforts.

### Features
* Real-time Classification: Analyze user-input text instantly to determine toxicity.
* NLP Pipeline: Incorporates tokenization, TF-IDF vectorization, and advanced pre-processing.
* Model Persistence: The model and vectorizer are serialized for fast loading and scalability.

### File Structure
* app.py: The Flask server and main application.
* toxicity_model.pkl: Pre-trained machine learning model for classification.
* tf_idf_vectorizer.pkl: TF-IDF vectorizer used for transforming text inputs.
* templates/index.html: HTML interface for users to input text and receive classification feedback.

### Installation

#### Prerequisites
* Python 3.8+
* Flask, Scikit-learn, and NLTK libraries

#### Steps
* Clone the repository:
git clone https://github.com/yourusername/toxic-comments-classification.git
* Navigate to the project directory:
cd toxic-comments-classification
* Install the dependencies:
pip install -r requirements.txt

### Usage
* Run the Flask application:
python app.py
* Access the app: Open a browser and go to http://localhost:5000.
* Usage on the App:
    * Enter a comment to classify.
    * Submit to see whether the text is labeled as toxic or non-toxic.
      
### Model Training
For training, this project uses the Toxic Tweets Dataset. To retrain or fine-tune the model, preprocess the dataset, apply TF-IDF vectorization, and use Scikit-learn's Naive Bayes or Logistic Regression classifiers.
