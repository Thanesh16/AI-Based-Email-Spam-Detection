# 🤖 AI-Powered Email Spam Detection and Classification System

An AI-powered email spam detection system that uses **Natural Language Processing (NLP)** and **Machine Learning** to classify emails as **Spam** or **Ham (legitimate)**.

The system preprocesses email content, converts text into numerical features using **TF-IDF**, and uses a trained **Linear Support Vector Machine (Linear SVM)** model to predict whether an email is spam.

---

## 📌 Project Overview

Email spam is one of the most common problems faced by internet users. Spam emails can contain unwanted advertisements, fraudulent offers, phishing attempts, malicious links, and other potentially harmful content.

This project aims to build an automated machine learning system that can analyze the content of an email and classify it as:

* 🟢 **Ham** — Legitimate email
* 🔴 **Spam** — Unwanted or suspicious email

The project combines **NLP text preprocessing**, **TF-IDF feature extraction**, and **supervised machine learning** to perform email classification.

---
## 🌐 Live Demo

🚀 **Try the application:**  
[AI Email Spam Detector](https://ai-email-spam-detector-kfqw.onrender.com)

The main objectives of this project are:

* Detect spam emails automatically.
* Apply NLP techniques to clean and preprocess email text.
* Convert unstructured email text into numerical features.
* Train and compare multiple machine learning classification models.
* Select the best-performing model based on evaluation metrics.
* Build a web-based interface for real-time email classification.
* Allow users to either paste email content or upload a `.txt` email file.
* Deploy the application as a web service.

---

## ✨ Features

### 📧 Email Classification

Classifies an email into:

```text
Spam
or
Ham
```

### 📝 Text Input

Users can paste email content directly into the web application.

### 📄 Email File Upload

Users can upload a `.txt` file containing an email message.

### 🧹 NLP Text Preprocessing

The application performs several preprocessing operations before classification:

* Lowercasing
* HTML tag removal
* URL removal
* Special character removal
* Extra whitespace removal
* Dataset-specific artifact removal
* English stopword removal

### 🔢 TF-IDF Feature Extraction

The cleaned email text is converted into numerical features using **TF-IDF (Term Frequency-Inverse Document Frequency)**.

### 🤖 Machine Learning Classification

Multiple machine learning algorithms were trained and evaluated:

* Logistic Regression
* Multinomial Naive Bayes
* Linear Support Vector Machine (Linear SVM)

### 🌐 Web Application

The trained model is integrated into a Flask web application with a responsive user interface.

---

# 🧠 System Architecture

The overall workflow of the system is:

```text
                Email Input
                    │
          ┌─────────┴─────────┐
          │                   │
     Text Input          .txt Upload
          │                   │
          └─────────┬─────────┘
                    ↓
           Text Preprocessing
                    ↓
        ┌───────────────────────┐
        │ Lowercase             │
        │ HTML Removal          │
        │ URL Removal           │
        │ Special Characters    │
        │ Extra Spaces          │
        │ Artifact Removal      │
        │ Stopword Removal      │
        └───────────────────────┘
                    ↓
             TF-IDF Vectorizer
                    ↓
              Linear SVM Model
                    ↓
          ┌─────────┴─────────┐
          ↓                   ↓
       🟢 Ham              🔴 Spam
```

---

# 📊 Dataset

The project uses an **email spam dataset based on the SpamAssassin Public Mail Corpus**.

The processed dataset contains:

* **5,640 emails**
* **2 columns**

  * `label`
  * `text`

### Label Encoding

```text
0 → Ham
1 → Spam
```

### Class Distribution

| Class     |     Count | Percentage |
| --------- | --------: | ---------: |
| Ham       |     4,072 |     72.20% |
| Spam      |     1,568 |     27.80% |
| **Total** | **5,640** |   **100%** |

The dataset is moderately imbalanced toward legitimate emails.

> **Dataset note:** The raw dataset is not included in this GitHub repository. The dataset is kept locally because it originates from an external email corpus. Please refer to the original dataset source and its applicable terms before redistributing the data.

---

# 🔍 Exploratory Data Analysis

Exploratory Data Analysis (EDA) was performed to understand the dataset before model training.

The analysis included:

* Dataset structure
* Missing-value checking
* Duplicate checking
* Class distribution
* Email length analysis
* Spam and Ham word-frequency analysis

### Dataset Quality

The dataset contains:

* **5,640 rows**
* **No missing values**
* **No duplicate rows**

### Email Length Analysis

Email character lengths were analyzed separately for Spam and Ham emails to understand their distribution and identify unusually long messages.

---

# 🧹 NLP Text Preprocessing

Raw email text contains HTML tags, URLs, punctuation, unnecessary spaces, and other elements that can introduce noise.

The following preprocessing pipeline was implemented.

## 1. Lowercasing

All text is converted to lowercase.

```text
"FREE MONEY NOW"
        ↓
"free money now"
```

This prevents words with different capitalization from being treated as different features.

---

## 2. HTML Removal

HTML tags are removed from email content.

```text
"<html>Hello</html>"
        ↓
"Hello"
```

---

## 3. URL Removal

URLs are removed from the email text.

```text
"Visit https://example.com now"
        ↓
"Visit now"
```

---

## 4. Special Character Removal

Unnecessary punctuation and special characters are removed while retaining letters, numbers, and spaces.

---

## 5. Extra Space Removal

Multiple spaces are converted into a single space.

---

## 6. Dataset Artifact Removal

Certain dataset-specific text artifacts such as:

```text
escapenumber
escapelong
```

are removed during preprocessing.

---

## 7. Stopword Removal

Common English words that generally provide limited classification information are removed using Scikit-learn's English stopword list.

Examples include words such as:

```text
the
is
and
to
of
```

---

# 🔢 TF-IDF Vectorization

Machine learning models cannot directly understand raw text.

Therefore, the cleaned email text is converted into numerical features using **TF-IDF**.

TF-IDF stands for:

> **Term Frequency – Inverse Document Frequency**

It gives greater importance to words that are useful for distinguishing documents while reducing the importance of words that occur frequently across many documents.

The project uses:

```python
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer()
X = tfidf.fit_transform(df['clean_text'])
```

The resulting feature matrix contains:

```text
5,640 emails
×
52,384 TF-IDF features
```

The trained TF-IDF vectorizer is saved and reused during prediction so that new emails are transformed using the **same feature representation used during model training**.

---

# 🤖 Machine Learning Models

Three supervised classification algorithms were evaluated.

## 1. Logistic Regression

Logistic Regression was used as one of the baseline classification models.

### Accuracy

**96%**

### Confusion Matrix

```text
[[812,   2],
 [ 45, 269]]
```

---

## 2. Multinomial Naive Bayes

Multinomial Naive Bayes is commonly used for text classification because it works well with word-frequency-based features.

### Accuracy

**92%**

---

## 3. Linear Support Vector Machine

Linear SVM was evaluated as another text classification model.

### Accuracy

**98%**

### Confusion Matrix

```text
[[808,   6],
 [ 19, 295]]
```

---

# 📈 Model Comparison

| Model                   | Accuracy |
| ----------------------- | -------: |
| Logistic Regression     |      96% |
| Multinomial Naive Bayes |      92% |
| **Linear SVM**          |  **98%** |

Based on the evaluation results, **Linear SVM was selected as the final model**.

---

# 🏆 Final Model Performance

The Linear SVM model achieved approximately:

| Metric    |  Ham | Spam |
| --------- | ---: | ---: |
| Precision | 0.98 | 0.98 |
| Recall    | 0.99 | 0.94 |
| F1-Score  | 0.98 | 0.96 |

Overall:

* **Accuracy:** 98%
* **Macro F1-score:** approximately 0.97
* **Weighted F1-score:** approximately 0.98

The model performed particularly well at identifying legitimate emails while also achieving strong spam detection performance.

---

# 🔬 Prediction Pipeline

When a user submits an email, the application follows this process:

```text
User enters email
        ↓
Preprocess email
        ↓
TF-IDF transformation
        ↓
Linear SVM prediction
        ↓
Numeric prediction
        ↓
1 → Spam
0 → Ham
```

For example:

```text
Prediction = 1
        ↓
Spam
```

or:

```text
Prediction = 0
        ↓
Ham
```

---

# 🌐 Web Application

The machine learning model is integrated into a **Flask web application**.

The interface provides:

### Email Content

Users can paste an email directly into a text area.

### Email File Upload

Users can upload an email stored as a `.txt` file.

### Detection

After submitting the email, the application displays the classification result.

```text
┌──────────────────────────────┐
│       AI Spam Detector       │
│                              │
│  Email Content               │
│  ┌────────────────────────┐  │
│  │ Paste email here...    │  │
│  └────────────────────────┘  │
│                              │
│            OR                │
│                              │
│       Upload .txt file       │
│                              │
│     [ 🔍 Detect Spam ]       │
│                              │
│       Analysis Result        │
│            Spam              │
└──────────────────────────────┘
```

---

# 🛠️ Technologies Used

## Programming Language

* Python

## Machine Learning

* Scikit-learn
* Linear SVM
* Logistic Regression
* Multinomial Naive Bayes

## Natural Language Processing

* Text preprocessing
* Stopword removal
* TF-IDF vectorization
* Regular expressions

## Data Analysis

* Pandas
* NumPy
* Matplotlib
* Seaborn

## Web Development

* Flask
* HTML
* CSS

## Model Serialization

* Python Pickle

## Deployment

* Gunicorn
* Render

---

# 📁 Project Structure

```text
AI-Based-Email-Spam-Detection/
│
├── app/
│   ├── app.py
│   │
│   ├── templates/
│   │   └── index.html
│   │
│   └── static/
│       └── style.css
│
├── data/
│   └── email_text.csv
│
├── model/
│   ├── spam_model.pkl
│   └── tfidf_vectorizer.pkl
│
├── notebooks/
│   ├── 01_data_understanding.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_text_preprocessing.ipynb
│   └── 04_model_training.ipynb
│
├── requirements.txt
├── .gitignore
└── LICENSE
```

> The `data/email_text.csv` dataset is intentionally not included in the public GitHub repository.

---

# ⚙️ Installation

## 1. Clone the repository

```bash
git clone https://github.com/Thanesh16/AI-Based-Email-Spam-Detection.git
```

## 2. Navigate into the project

```bash
cd AI-Based-Email-Spam-Detection
```

## 3. Create a virtual environment

```bash
python -m venv venv
```

## 4. Activate the virtual environment

### Windows

```bash
venv\Scripts\activate
```

### macOS / Linux

```bash
source venv/bin/activate
```

## 5. Install dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Running the Application Locally

From the project root directory:

```bash
python app/app.py
```

The Flask application will start locally.

Open the local URL displayed in the terminal, typically:

```text
http://127.0.0.1:5000/
```

---

# 📧 How to Use

### Method 1 — Paste Email

1. Open the application.
2. Paste an email into the **Email Content** box.
3. Click **Detect Spam**.
4. View the prediction.

### Method 2 — Upload Email File

1. Create or select a `.txt` email file.
2. Upload the file.
3. Click **Detect Spam**.
4. View the prediction.

---

# 🧪 Example

### Spam Email

```text
Subject: Congratulations! You Have Won $1,000,000

Congratulations!

You have been selected as the lucky winner of our special cash prize.

Click the link below to claim your prize immediately.

Act now to receive your reward!
```

Expected classification:

```text
🔴 Spam
```

### Ham Email

```text
Subject: Meeting Tomorrow

Hi John,

Just a reminder that our project meeting is scheduled for tomorrow at 10:00 AM in the conference room.

Please bring the latest project report and your presentation slides.

Thanks,
David
```

Expected classification:

```text
🟢 Ham
```

---

# 🚀 Deployment

The application is structured for deployment as a Flask web service.

The production application uses **Gunicorn** as the WSGI server.

The deployment start command is:

```bash
gunicorn app.app:app
```

The model and TF-IDF vectorizer are loaded using project-relative paths, allowing the application to work without hardcoded local Windows paths.

---

# 📦 Saved Machine Learning Artifacts

The trained machine learning components are stored in the `model/` directory.

### `spam_model.pkl`

Contains the trained **Linear SVM** model used for spam/ham prediction.

### `tfidf_vectorizer.pkl`

Contains the trained **TF-IDF vectorizer** used to transform new email text into numerical features.

Both files are required for the deployed application.

---

# 🔐 Important Dataset Note

The raw email dataset is not included in this repository.

The project uses an external email spam corpus, and the raw data may contain email content or other information that should not be redistributed without checking the applicable dataset terms.

The trained model and application code are provided separately from the raw dataset.

---

# ⚠️ Limitations

Although the model achieved 98% accuracy on the test set, several limitations remain:

* The model relies primarily on email text content.
* It does not inspect attachments.
* It does not analyze sender reputation.
* It does not perform URL reputation checks.
* It does not detect malicious attachments.
* Real-world emails may differ from the training dataset.
* Dataset-specific artifacts and preprocessing choices may affect predictions.
* Model performance can change when applied to emails from a different distribution.

Therefore, the prediction should be treated as an automated classification result rather than a guarantee that an email is safe or malicious.

---

# 🔮 Future Improvements

Possible future improvements include:

* Add confidence/probability estimation using a calibrated classifier.
* Support additional email file formats.
* Analyze email headers and sender information.
* Extract and analyze URLs.
* Detect phishing-specific patterns.
* Add attachment analysis.
* Experiment with advanced NLP models such as BERT.
* Improve preprocessing for different email formats.
* Add a database for prediction history.
* Add authentication and user accounts.
* Monitor model performance on new email data.
* Implement automated model retraining.

---

# 📚 Notebooks

The project includes separate notebooks for different stages of development.

### `01_data_understanding.ipynb`

Dataset inspection and initial understanding.

### `02_eda.ipynb`

Exploratory Data Analysis including class distribution, email length, and word-frequency analysis.

### `03_text_preprocessing.ipynb`

NLP preprocessing and TF-IDF vectorization.

### `04_model_training.ipynb`

Model training, evaluation, comparison, and final model selection.

---

# 🎓 Key Concepts Demonstrated

This project demonstrates practical implementation of:

* Natural Language Processing
* Text preprocessing
* Regular expressions
* Stopword removal
* TF-IDF
* Feature extraction
* Supervised Machine Learning
* Logistic Regression
* Naive Bayes
* Linear SVM
* Train/Test Split
* Stratified sampling
* Classification metrics
* Confusion Matrix
* Model serialization
* Flask
* Web application development
* Machine Learning deployment

---

# 📌 Conclusion

This project demonstrates an end-to-end **Natural Language Processing and Machine Learning pipeline** for email spam detection.

The system takes raw email text, preprocesses it, converts it into TF-IDF features, and uses a trained Linear SVM model to classify the email as Spam or Ham.

Among the evaluated models, **Linear SVM achieved the best performance with approximately 98% accuracy**, making it the selected model for the final application.

The project also demonstrates how a machine learning model can be integrated into a practical Flask web application and prepared for deployment.

---

# 👨‍💻 Connect With Me

**Thanesh S**

* GitHub: [Thanesh16](https://github.com/Thanesh16)
* LinkedIn: [*Click Here*](https://linkedin.com/in/thanesh006)
* Email: [Click Here](mailto:thaneshselvam4@gmail.com)

---

## ⭐ If you find this project useful

Feel free to explore the repository, learn from the implementation, and provide feedback.
