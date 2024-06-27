import streamlit as st
from streamlit_chat import message
import tempfile
import os
# from langchain.document_loaders.csv_loader import CSVLoader
from langchain_community.document_loaders import CSVLoader
# from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain.vectorstores import FAISS
from langchain_community.vectorstores import FAISS
# from langchain.llms.ctransformers import CTransformers
from langchain_community.llms.ctransformers import CTransformers
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain.vectorstores.base import VectorStoreRetriever
from transformers import AutoModelForMaskedLM, AutoModelForCausalLM, AutoTokenizer
from langchain.prompts import PromptTemplate

from langchain.llms.base import LLM
from typing import Any, List, Optional

class HuggingFaceLLM(LLM):
    model: Any
    tokenizer: Any

    def __init__(self, model, tokenizer):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt")
        output = self.model.generate(**inputs, max_new_tokens=50)
        return self.tokenizer.decode(output[0], skip_special_tokens=True)

    @property
    def _llm_type(self) -> str:
        return "custom_huggingface_llm"


DB_FAISS_PATH = "vectorestore/db_faiss"

# loading the model
# def load_llm():
#     llm = CTransformers(
#         model = "llama-2-7b-chat.ggmlv3.q4_K_S.bin",
#         model_type = "llama",
#         max_new_tokens = 512,
#         temperature = 0.5
#     )

#     return llm

# def load_llm() :
#     tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
#     model = AutoModelForMaskedLM.from_pretrained("distilbert-base-uncased")
#     return model, tokenizer

def load_llm():
    model_name = "microsoft/DialoGPT-small"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    llm = HuggingFaceLLM(model=model, tokenizer=tokenizer)

    prompt_template = """Context: {context}

    Human: {question}
    Assistant: Based on the context, here's a relevant quote:"""

    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    return llm, PROMPT

st.title("Chat with CSV")

# st.markdown("<h3 style='text-align: center'>Build by Ninjavin</h3>", unsafe_allow_html=True)

uploaded_file = st.sidebar.file_uploader("Upload your CSV Data!", type="csv")

print(uploaded_file)

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name
        print("tmp file path : " + tmp_file_path)

    loader = CSVLoader(file_path=tmp_file_path, encoding="utf-8", csv_args={
        "delimiter": ','
    })

    data = loader.load()
    # print("data : " + data)
    st.json(data)

    embeddings = HuggingFaceEmbeddings(
        model_name='sentence-transformers/all-MiniLM-L6-v2',
        model_kwargs = {'device': 'cpu'}
    )

    db = FAISS.from_documents(data, embeddings)
    db.save_local(DB_FAISS_PATH)

    llm, prompt = load_llm()
    retriever = VectorStoreRetriever(vectorstore=db)
    chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever, combine_docs_chain_kwargs={"prompt" : prompt}, return_source_documents=True)

    def conversational_chat(query):
        result = chain({"question": query, "chat_history": st.session_state['history']})
        answer = result["answer"]
        
        # Clean up the answer
        answer = answer.strip()
        
        # If the answer is empty or just repeats the prompt, try to extract a quote directly from the context
        if not answer or answer == "Based on the context, here's a relevant quote:":
            context = result.get('source_documents', [])
            if context:
                # Extract quotes from the context
                quotes = [doc.page_content for doc in context if 'Quote:' in doc.page_content]
                if quotes:
                    answer = "Here's a relevant quote: " + quotes[0].split('Quote:')[1].split('Character:')[0].strip()
                else:
                    answer = "I couldn't find a specific quote, but here's some relevant information: " + context[0].page_content
        
        st.session_state['history'].append((query, answer))
        return answer
    
    
    
    if 'history' not in st.session_state:
        st.session_state['history'] = []

    if 'generated' not in st.session_state:
        st.session_state['generated'] = ["Hello, Ask me anything about your uploaded file!"]

    if 'past' not in st.session_state:
        st.session_state['past'] = ["Hey! How you doing?"]
    
    # container for chat history
    response_container = st.container()
    container = st.container()

    print("Before container!")

    with container:
        with st.form(key="my_form", clear_on_submit=True):
            user_input = st.text_input("Query : ", placeholder="Chat with your CSV...", key='input')
            submit_button = st.form_submit_button(label = "Chat")
            print(user_input)
            print(submit_button)

        if submit_button and user_input:
            output = conversational_chat(user_input)
            # print("Output : " + output)
            st.session_state['past'].append(user_input)
            st.session_state['generated'].append(output)

    if st.session_state['generated']:
        with response_container:
            for i in range(len(st.session_state['generated'])):
                message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style='big-smile')
                message(st.session_state["generated"][i], key=str(i), avatar_style="thumbs")

else:
    print("File not uploaded!")