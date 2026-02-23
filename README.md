
# 📰 Fake News Detection using NLP & Machine Learning

## 📌 Project Overview

This project focuses on detecting whether a news article is **Real** or **Fake** using **Natural Language Processing (NLP)** and **Machine Learning techniques**.

The system processes news text, converts it into numerical features, and trains a classification model to automatically identify misleading or false information.

---

## 🎯 Objective

To build a machine learning model that can:

* ✅ Classify news as **Real**
* ❌ Classify news as **Fake**
* Improve detection accuracy using text preprocessing and feature extraction

---

## 🛠 Technologies Used

* Python
* Pandas
* NumPy
* Scikit-Learn
* NLTK
* Matplotlib / Seaborn
* Jupyter Notebook

---

## 📂 Dataset

The dataset contains:

* News title
* News text/content
* Label (Real / Fake)

Data preprocessing includes:

* Removing punctuation
* Lowercasing text
* Removing stopwords
* Tokenization
* Vectorization (TF-IDF)

---

## 🧠 Model Workflow

1. **Data Loading**
2. **Text Cleaning & Preprocessing**
3. **Feature Extraction (TF-IDF)**
4. **Train-Test Split**
5. **Model Training**
6. **Model Evaluation**
7. **Prediction**

---

## 🤖 Machine Learning Models Used

* Logistic Regression
* Naive Bayes
* Passive Aggressive Classifier
* Support Vector Machine (if included)

---

## 📊 Evaluation Metrics

* Accuracy
* Confusion Matrix
* Precision
* Recall
* F1-Score

---

## 🚀 How to Run the Project

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/yourusername/fake-news-detection.git
cd fake-news-detection
```

### 2️⃣ Install Dependencies

```bash
pip install pandas numpy scikit-learn nltk matplotlib seaborn
```

### 3️⃣ Run Notebook

```bash
jupyter notebook
```

Open:

```
fake_new_detection.ipynb
```

Run all cells to train and evaluate the model.

---

## 📈 Sample Output

The model predicts whether a given news article is:

```
Input: "Breaking news: Government announces..."
Output: REAL
```

or

```
Input: "Shocking secret cure discovered..."
Output: FAKE
```

---

## 🔥 Future Improvements

* Use Deep Learning (LSTM / BERT)
* Deploy as a Web Application
* Add Real-time News API Integration
* Improve dataset size for better accuracy

---

## 🏆 Key Learnings

* Text preprocessing in NLP
* Feature extraction using TF-IDF
* Binary text classification
* Model evaluation techniques
* Handling imbalanced datasets

---

