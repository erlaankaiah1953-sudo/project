import streamlit as st

from src.rag_chain import build_rag_chain


@st.cache_resource
def get_rag_chain():
    return build_rag_chain()


st.set_page_config(
    page_title="Tasty Bites Assistant",
    page_icon="🍽️",
    layout="centered",
)

st.title("Tasty Bites")
st.caption("Ask me about our menu, prices, and restaurant information.")

with st.sidebar:
    st.header("Restaurant assistant")
    st.write(
        "Get answers based on the restaurant knowledge base. "
        "For allergy concerns, confirm ingredients directly with the restaurant."
    )
    if st.button("Clear conversation", use_container_width=True):
        st.session_state.messages = []

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

question = st.chat_input("Ask about our restaurant...")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    try:
        with st.chat_message("assistant"):
            with st.spinner("Finding an answer..."):
                answer = get_rag_chain().invoke({"question": question})
            st.markdown(answer)
        st.session_state.messages.append(
            {"role": "assistant", "content": answer}
        )
    except Exception as error:
        st.error(f"Unable to get an answer: {error}")
