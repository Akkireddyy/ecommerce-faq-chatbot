# 🛒 AI-Powered E-Commerce FAQ Chatbot
An AI-powered FAQ chatbot built using Python, NLP, TF-IDF vectorization, cosine similarity, and Streamlit.
The chatbot answers customer queries related to e-commerce topics such as orders, refunds, payments, delivery, returns, and account management.



# 📌 Features
* Interactive chatbot UI using Streamlit
* FAQ dataset handling using CSV
* NLP text preprocessing using NLTK
* Tokenization and stopword removal
* Stemming for improved text matching
* TF-IDF vectorization
* Cosine similarity-based FAQ matching
* Chat history support
* Typing animation effect
* Custom chatbot styling and chat bubbles
* Clear chat functionality



# 🛠️ Technologies Used
* Python
* Streamlit
* NLTK
* scikit-learn
* pandas


# 📖 How It Works
1. User enters a question in the chatbot.
2. The input text is preprocessed using NLP techniques:

   * Lowercasing
   * Punctuation removal
   * Tokenization
   * Stopword removal
   * Stemming
3. The processed text is converted into TF-IDF vectors.
4. Cosine similarity is calculated between the user query and stored FAQs.
5. The chatbot returns the most relevant answer.



# ▶️ Installation & Setup

## 1. Clone the Repository
```bash
git clone <repository-link>
```



## 2. Install Required Libraries
```bash
python -m pip install -r requirements.txt
```



## 3. Run the Application
```bash
python -m streamlit run app.py
```


# 📷 Sample Features
* FAQ Question Matching
* Real-Time Chat Interface
* AI-Based Response Retrieval
* E-Commerce Support Assistant


# 📌 Example Questions
* How can I track my order?
* What is your refund policy?
* How do I cancel my order?
* Can I pay using UPI?
* How do I return a product?



# 📈 Future Improvements
* Voice input support
* Multi-language support
* Database integration
* Deep learning chatbot integration
* User authentication system



# 👨‍💻 Developed By
Akshaya Reddy
