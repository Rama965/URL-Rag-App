import streamlit as st
from rag import load_url, split_documents, create_vector_db, build_rag_chain

# -------------------- PAGE CONFIG --------------------
st.set_page_config(
    page_title="ChatWebRAG",
    page_icon="💬",
    layout="centered"
)

# -------------------- SUBTLE CHAT CSS --------------------
st.markdown("""
<style>
html, body, [class*="css"] {
    font-family: Inter, -apple-system, BlinkMacSystemFont, sans-serif;
}

.block-container {
    max-width: 760px;
    padding-top: 1.5rem;
}

.chat-user {
    background-color: #e8ebf8;
    padding: 12px 16px;
    border-radius: 14px;
    margin: 10px 0 10px auto;
    max-width: 80%;
}

.chat-ai {
    background-color: #f7f7f8;
    padding: 14px 18px;
    border-radius: 14px;
    margin: 10px auto 10px 0;
    max-width: 80%;
    border-left: 3px solid #4f46e5;
}

.system-box {
    background: #fafafa;
    padding: 14px;
    border-radius: 10px;
    border: 1px dashed #dcdcdc;
    margin-bottom: 1rem;
}
</style>
""", unsafe_allow_html=True)

# -------------------- HEADER --------------------
st.title("ChatWebRAG 💬")
st.caption("Chat with any website. Context-aware. No hallucinations.")

# -------------------- INIT SESSION --------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "chain" not in st.session_state:
    st.session_state.chain = None

# -------------------- URL INPUT --------------------
with st.expander("🌐 Load Website", expanded=True):
    url = st.text_input("Website URL", placeholder="https://example.com")

    if st.button("Load & Index"):
        if not url:
            st.warning("Please enter a valid URL.")
        else:
            with st.spinner("Indexing website content..."):
                docs = load_url(url)
                chunks = split_documents(docs)
                db = create_vector_db(chunks)
                st.session_state.chain = build_rag_chain(db)
                st.session_state.messages = []

            st.success("Website loaded. Start chatting below.")

# -------------------- CHAT AREA --------------------
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f"<div class='chat-user'>{msg['content']}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='chat-ai'>{msg['content']}</div>", unsafe_allow_html=True)

# -------------------- INPUT --------------------
if st.session_state.chain:
    user_input = st.chat_input("Ask something about the website...")

    if user_input:
        # User message
        st.session_state.messages.append({
            "role": "user",
            "content": user_input
        })

        with st.spinner("Thinking..."):
            response = st.session_state.chain.invoke(user_input)

        # AI message
        st.session_state.messages.append({
            "role": "assistant",
            "content": response.content
        })

        st.rerun()
else:
    st.info("👆 Load a website to start chatting.")

# -------------------- FOOTER --------------------
st.divider()
st.caption("LangChain • Groq • Chroma • Streamlit")
