# ============================================================
# 1. IMPORT REQUIRED LIBRARIES
# ============================================================

from flask import Flask, render_template, request
import pickle
import re
import os
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


# ============================================================
# 2. CREATE FLASK APPLICATION
# ============================================================

# Creates the Flask application
app = Flask(__name__)


# ============================================================
# 3. SET PROJECT PATHS
# ============================================================

# Find the main project folder
# app.py is inside the "app" folder,
# so we go one level up to reach the project folder.

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

# Path to the trained Linear SVM model
MODEL_PATH = os.path.join(
    BASE_DIR,
    "model",
    "spam_model.pkl"
)

# Path to the TF-IDF vectorizer
TFIDF_PATH = os.path.join(
    BASE_DIR,
    "model",
    "tfidf_vectorizer.pkl"
)


# ============================================================
# 4. LOAD TRAINED MACHINE LEARNING MODEL
# ============================================================

# Load the trained Linear SVM model
# This model is responsible for predicting:
# 0 = Ham
# 1 = Spam

with open(MODEL_PATH, "rb") as file:
    model = pickle.load(file)


# ============================================================
# 5. LOAD TF-IDF VECTORIZER
# ============================================================

# Load the TF-IDF vectorizer that was created during training.
# It converts email text into numerical features
# that the machine learning model can understand.

with open(TFIDF_PATH, "rb") as file:
    tfidf = pickle.load(file)


# ============================================================
# 6. TEXT PREPROCESSING FUNCTION
# ============================================================

# This function performs the SAME text preprocessing
# that we used while training the machine learning model.

def preprocess_text(text):

    # Convert all text to lowercase
    text = text.lower()

    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)

    # Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)

    # Remove special characters
    # Keep only letters, numbers and spaces
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)

    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()

    # Remove dataset-specific artifacts
    text = re.sub(r'escapenumber\w*', '', text)
    text = re.sub(r'escapelong', '', text)

    # Split the text into individual words
    words = text.split()

    # Remove English stopwords
    words = [
        word for word in words
        if word not in ENGLISH_STOP_WORDS
    ]

    # Join the remaining words back into a sentence
    return ' '.join(words)


# ============================================================
# 7. EMAIL PREDICTION FUNCTION
# ============================================================

# This function takes an email as input
# and returns either "Spam" or "Ham".

def predict_email(text):

    # Preprocess the email
    clean_text = preprocess_text(text)

    # Convert the cleaned email into TF-IDF numerical features
    text_vector = tfidf.transform([clean_text])

    # Use the trained model to make a prediction
    prediction = model.predict(text_vector)[0]

    # Convert numerical prediction into readable result
    if prediction == 1:
        return "Spam"
    else:
        return "Ham"


# ============================================================
# 8. HOME ROUTE
# ============================================================

# This route handles both:
# GET  -> displaying the webpage
# POST -> receiving the email entered by the user

@app.route("/", methods=["GET", "POST"])
def home():

    # Stores the prediction result
    result = None

    # Stores error messages
    error = None

    # Execute this section only when the user submits the form
    if request.method == "POST":

        # Get email text entered/pasted by the user
        email_text = request.form.get("email_text", "")

        # Get uploaded file
        uploaded_file = request.files.get("email_file")


        # --------------------------------------------------------
        # 8.1 CHECK TYPED / PASTED EMAIL
        # --------------------------------------------------------

        if email_text.strip():

            # Predict whether the email is Spam or Ham
            result = predict_email(email_text)


        # --------------------------------------------------------
        # 8.2 CHECK UPLOADED FILE
        # --------------------------------------------------------

        elif uploaded_file and uploaded_file.filename:

            # Allow only .txt files
            if not uploaded_file.filename.lower().endswith(".txt"):

                error = "Please upload a .txt email file."

            else:

                # Read the uploaded file
                email_text = uploaded_file.read().decode("utf-8")

                # Check whether the file contains any text
                if email_text.strip():

                    # Predict whether the email is Spam or Ham
                    result = predict_email(email_text)

                else:

                    # Show error if the file is empty
                    error = "The uploaded file is empty."


        # --------------------------------------------------------
        # 8.3 NOTHING WAS PROVIDED
        # --------------------------------------------------------

        else:

            error = "Please enter an email or upload a file."


    # Send the result/error to the HTML page
    return render_template(
        "index.html",
        result=result,
        error=error
    )


# ============================================================
# 9. RUN FLASK APPLICATION
# ============================================================

# This starts the Flask development server
# when we run app.py directly.

if __name__ == "__main__":
    app.run()