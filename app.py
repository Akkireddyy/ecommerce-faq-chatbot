import streamlit as st
import pandas as pd
import nltk
import string
import time

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download NLTK resources
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

# Stemmer
stemmer = PorterStemmer()

# Load FAQ dataset
df = pd.read_csv("faqs.csv")

# Text preprocessing function
def preprocess_text(text):

    # Convert to lowercase
    text = text.lower()

    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Tokenization
    words = word_tokenize(text)

    # Remove stopwords + stemming
    filtered_words = [
        stemmer.stem(word)
        for word in words
        if word not in stopwords.words('english')
    ]

    return " ".join(filtered_words)

# Preprocess FAQ questions
df["processed_question"] = df["question"].apply(preprocess_text)

# TF-IDF Vectorization
vectorizer = TfidfVectorizer()

faq_vectors = vectorizer.fit_transform(df["processed_question"])

# Chatbot response function
def get_chatbot_response(user_input):

    # Preprocess user input
    processed_input = preprocess_text(user_input)

    # Convert user input to vector
    user_vector = vectorizer.transform([processed_input])

    # Calculate cosine similarity
    similarity_scores = cosine_similarity(
        user_vector,
        faq_vectors
    )

    # Find best matching FAQ
    best_match_index = similarity_scores.argmax()

    best_score = similarity_scores[0][best_match_index]

    # Similarity threshold
    if best_score < 0.3:
        return (
            "Sorry, I couldn't find a relevant answer. "
            "Please ask about orders, refunds, delivery or payments."
        )

    # Return best answer
    return df.iloc[best_match_index]["answer"]

# ---------------- STREAMLIT UI ----------------

st.set_page_config(
    page_title="E-Commerce FAQ Chatbot",
    page_icon="🤖",
    layout="centered"
)
st.markdown("""
<style>

/* Chat container outline */
.chat-container {
    border: 1px solid #d3d3d3;
    border-radius: 12px;
    padding: 15px;
    margin-top: 10px;
    background-color: white;
}



</style>
""", unsafe_allow_html=True)



# Logo
col1, col2, col3 = st.columns([3,2,2])

with col2: 
 st.image("assets/chatbot.png", width=130)

# Title
st.markdown(
    "<h1 style='text-align:center;'>🛒AI-Powered E-Commerce FAQ Chatbot</h1>",
    unsafe_allow_html=True
)



# ---------------- SIDEBAR ----------------

st.sidebar.title("Welcome")
col1, col2 = st.columns([150,30])

with col2:

    if st.button("Clear chat"):

        st.session_state.messages = []

        st.rerun()

st.sidebar.info(
    """
   Ask questions related to orders, refunds, payments and delivery🛒 🛍️.
    """
)

st.sidebar.subheader("Sample Questions")

st.sidebar.write("- How can I track my order🚚?")
st.sidebar.write("- What is your refund policy💸?")
st.sidebar.write("- How do I cancel my order❌?")
#st.sidebar.image("assets/chatbot.png", width=120)


# Initialize chat history
if "messages" not in st.session_state:

    st.session_state.messages = []

# Welcome message
if len(st.session_state.messages) == 0:

    st.session_state.messages.append({
        "role": "assistant",
        "content":
        "Hello 👋 I am your e-commerce support assistant."
    })

# DISPLAY OLD CHAT HISTORY FIRST
for message in st.session_state.messages:

    if message["role"] == "user":
       st.markdown(
        f"""
    <div style="
        background-color:#2563EB;
        color:white;
        padding:10px 15px;
        border-radius:15px;
        margin:10px;
        width:fit-content;
        margin-left:auto;
        max-width:70%;
        border:1px solid #555555;
    ">
    {message['content']}
    </div>
    """,
    unsafe_allow_html=True
    )


    else:

      st.markdown(
    f"""
    <div style="
        background-color:#374151;
        color:white;
        padding:10px 15px;
        border-radius:15px;
        margin:10px;
        width:fit-content;
        margin-right:auto;
        max-width:70%;
        border:1px solid #555555;
    ">
    {message['content']}
    </div>
    """,
    unsafe_allow_html=True
)

# CHAT INPUT
user_question = st.chat_input("Type your question here...")

# NEW MESSAGE
if user_question:

    # Save user message
    st.session_state.messages.append({
        "role": "user",
        "content": user_question
    })

    # Bot response
    with st.spinner("Typing..."):

        time.sleep(1)

        response = get_chatbot_response(user_question)

    # Save bot response
    st.session_state.messages.append({
        "role": "assistant",
        "content": response
    })

    # Rerun page to refresh full chat
    st.rerun()
    # Save assistant response
    st.session_state.messages.append({
        "role": "assistant",
        "content": response
    })
    
      