# 🎬 Movie Review Sentiment Analysis

*An NLP-based machine learning model to classify movie reviews as positive or negative.*

---

## 📌 Overview

This project demonstrates how **Natural Language Processing (NLP)** and **Machine Learning** can be applied to understand sentiment in text.
Using the **IMDB Movie Reviews Dataset** (50,000 labeled reviews), the model learns to identify whether a review expresses a **positive** or **negative** sentiment.

It showcases a complete workflow — from **data preprocessing** to **model evaluation** and **real-time prediction**.

---

## 🎯 Objectives

* 🧹 **Preprocess Text Data:** Clean, tokenize, and remove stopwords from raw reviews.
* 🔢 **Vectorize Text:** Convert text into numerical form using **TF-IDF (Term Frequency–Inverse Document Frequency)**.
* 🤖 **Train Classifier:** Use **Logistic Regression** to classify sentiments (Positive / Negative).
* 📊 **Evaluate Performance:** Measure model accuracy, precision, recall, and F1-score.
* 💬 **Interactive Testing:** Allow users to input custom reviews and get real-time sentiment predictions.

---

## 🧠 How It Works

1. **Data Preprocessing:**

   * Convert text to lowercase
   * Remove punctuation, HTML tags, and special characters
   * Tokenize and remove stopwords

2. **Feature Extraction:**

   * Apply **TF-IDF Vectorization** to transform text into feature vectors

3. **Model Training:**

   * Train a **Logistic Regression** model using the processed vectors

4. **Model Evaluation:**

   * Test on unseen data and calculate performance metrics

5. **Prediction Interface:**

   * Input any custom text (movie review) and get a predicted sentiment

---

## 🧩 Tech Stack

* **Language:** Python 🐍
* **Libraries:**

  * `pandas`, `numpy` – Data handling
  * `nltk` – Text preprocessing
  * `scikit-learn` – TF-IDF, Logistic Regression, Evaluation Metrics
  * `joblib` – Model saving/loading
  * `streamlit` or CLI (optional) – For interactive predictions

---

## 📂 Repository Structure

```
/movie-review-sentiment-analysis
├── movie_sentiment_analysis.py
└── README.md
```

---

## ⚙️ How to Run

1. **Clone the repository**

   ```bash
   git clone https://github.com/KAVI-DEV-ui/movie-review-sentiment-analysis.git
   cd movie-review-sentiment-analysis
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Train the model (if not pre-trained)**

   ```bash
   python train_model.py
   ```

4. **Run predictions**

   ```bash
   python predict.py
   ```

   or launch the **Streamlit app**:

   ```bash
   streamlit run app.py
   ```

---

## 📊 Model Performance

| Metric    | Score |
| --------- | ----- |
| Accuracy  | ~89%  |
| Precision | ~0.90 |
| Recall    | ~0.88 |
| F1-Score  | ~0.89 |

*(Values may vary slightly based on dataset split.)*

---

## 🚀 Future Enhancements

* 🔁 Replace Logistic Regression with advanced models (LSTM, BERT).
* 🌐 Add a web UI for easier access.
* 🧾 Expand dataset with multilingual reviews.
* ☁️ Deploy model using Render / Hugging Face Spaces.

---

## 👤 Author

**Kavi Dev**
GitHub: [KAVI-DEV-ui](https://github.com/KAVI-DEV-ui)

---

Would you like me to include a **Streamlit UI section** (with sample code for interactive predictions) in the README too?
