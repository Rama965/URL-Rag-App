import streamlit as st
from rag import load_url, split_documents, create_vector_db, build_rag_chain

# -------------------- PAGE CONFIG --------------------
st.set_page_config(
    page_title="ChatWebRAG",
    page_icon="💬",
    layout="centered"
)

# -------------------- GLOBAL STYLE --------------------
st.markdown("""
<style>
html, body, [class*="css"] {
    font-family: Inter, -apple-system, BlinkMacSystemFont, sans-serif;
}
.block-container {
    max-width: 760px;
    padding-top: 1.5rem;
}
</style>
""", unsafe_allow_html=True)

# -------------------- HEADER --------------------
st.title("ChatWebRAG 💬")
st.caption("Chat with any website. Context-aware. No hallucinations.")

# -------------------- SESSION STATE --------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "chain" not in st.session_state:
    st.session_state.chain = None

# -------------------- URL LOADER --------------------
with st.expander("🌐 Load Website", expanded=True):
    url = st.text_input(
        "Website URL",
        placeholder="https://example.com"
    )

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

            st.success("Website loaded successfully. Start chatting below 👇")

# -------------------- CHAT HISTORY --------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# -------------------- CHAT INPUT --------------------
if st.session_state.chain:
    user_input = st.chat_input("Ask something about the website...")

    if user_input:
        # User message
        st.session_state.messages.append({
            "role": "user",
            "content": user_input
        })

        with st.chat_message("user"):
            st.markdown(user_input)

        # AI response
        with st.spinner("Thinking..."):
            response = st.session_state.chain.invoke(user_input)
            ai_reply = response.content

        st.session_state.messages.append({
            "role": "assistant",
            "content": ai_reply
        })

        with st.chat_message("assistant"):
            st.markdown(ai_reply)
else:
    st.info("👆 Load a website to start chatting.")

# -------------------- FOOTER --------------------
st.divider()
st.caption("⚡ LangChain • Groq • Chroma • Streamlit")
