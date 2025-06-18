import streamlit as st
import os
from dotenv import load_dotenv
import pandas as pd
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI
load_dotenv()
st.set_page_config(page_title="⚽ Football Genius Chatbot", layout="centered")
st.title("⚽ Football Genius Chatbot")
st.write("I'm your AI teammate! Ask me anything about football stats, players, tactics, or match insights.")
@st.cache_resource
def initialize_qa_chain():
    try:
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            st.error("GOOGLE_API_KEY not found in environment variables. Please set it.")
            return None, None        
        df = pd.read_csv("football_data.csv")        
        df['Question'] = df['Question'].str.strip().str.lower()
        df['Answer'] = df['Answer'].fillna("No answer available.")        
        docs = [
            Document(page_content=f"Question: {row['Question']} Answer: {row['Answer']}")
            for _, row in df.iterrows()
        ]
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vectorstore = FAISS.from_documents(docs, embeddings)
        retriever = vectorstore.as_retriever()
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.4)
        qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
        return qa_chain, df
    except Exception as e:
        st.error(f"Error initializing QA chain: {e}")
        return None, None
qa_chain, qa_dataframe = initialize_qa_chain()
if "messages" not in st.session_state:
    st.session_state.messages = []
if "preset_question" not in st.session_state:
    st.session_state.preset_question = None
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
st.markdown("#### ⚡ Try asking:")
col1, col2, col3 = st.columns(3)
with col1:
    if st.button("What is xG in football?"):
        st.session_state.preset_question = "what is expected goals (xg) in football?"
with col2:
    if st.button("Top Premier League assisters"):
        st.session_state.preset_question = "who has the most assists in the premier league?"
with col3:
    if st.button("Explain me your model"):
        st.session_state.preset_question = "explain me your model"
prompt = st.chat_input("Ask your football question...")
final_prompt = None
if st.session_state.preset_question:
    final_prompt = st.session_state.preset_question
    st.session_state.preset_question = None
elif prompt:
    final_prompt = prompt
if final_prompt and qa_chain:
    st.session_state.messages.append({"role": "user", "content": final_prompt})
    with st.chat_message("user"):
        st.markdown(final_prompt)
    normalized_prompt = final_prompt.strip().lower()
    with st.spinner("Thinking..."):
        try:
            # Step 1: Exact match from CSV
            direct_match = qa_dataframe[qa_dataframe['Question'] == normalized_prompt]
            if not direct_match.empty:
                result = direct_match.iloc[0]['Answer']
            else:
                # Step 2: Fallback to semantic search
                response = qa_chain.invoke(normalized_prompt)
                result = response.get("result", "Sorry, I couldn’t find an answer.")
        except Exception as e:
            result = f"Error: {e}. Please try again."
    st.session_state.messages.append({"role": "assistant", "content": result})
    with st.chat_message("assistant"):
        st.markdown(result)
st.markdown("---")
st.caption("⚽ Powered by Google Gemini, LangChain, and real football data.")
