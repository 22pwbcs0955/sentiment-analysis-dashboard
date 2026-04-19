# 📝 Sentiment Analysis Dashboard

An interactive **Sentiment Analysis project** that classifies text as **Negative, Neutral, or Positive** using **TF-IDF + Logistic Regression**. Built with Python and deployed as a **Streamlit dashboard**.

[![Python](https://img.shields.io/badge/python-3.10+-blue)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-ready-brightgreen)](https://streamlit.io/)
[![License](https://img.shields.io/badge/license-MIT-yellow.svg)](LICENSE)

---

## ✨ Features
- 🟢 Multi-class sentiment classification (Negative / Neutral / Positive)  
- 📊 Confidence scores for each prediction  
- ☁️ Word cloud visualization of input text  
- 💻 Interactive Streamlit dashboard for real-time predictions  

---

## 🚀 Usage

### 1️⃣ Train the model
Run `main.py` to preprocess data, train the Logistic Regression model, and save the model/vectorizer:
python main.py
### 2️⃣ Launch the dashboard
streamlit run app.py
Paste text into the input box
Click Analyze
View predicted sentiment, confidence scores, and word cloud
📂 Dataset
Original dataset: Sentiment140 on Kaggle
Place CSV in data/ folder before running main.py.
⚙️ Dependencies
Python 3.10+
pandas, numpy, scikit-learn, nltk
matplotlib, seaborn, wordcloud, streamlit

You can generate requirements.txt using:

pip freeze > requirements.txt
💡 Notes
Large datasets and trained models are not included.
Dashboard dynamically handles the classes your model was trained on.

### 📜 License

MIT License
rectly to the **main branch**.
