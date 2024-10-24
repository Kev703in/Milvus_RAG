import streamlit as st
from streamlit_chat import message
import milvusInjest
from pymilvus import connections
from langchain_community.vectorstores import Milvus
from langchain_community.embeddings import HuggingFaceEmbeddings
from accelerate import Accelerator
from langchain.prompts import PromptTemplate
from langchain_community.llms import CTransformers
from langchain.chains import RetrievalQA
from milvusConnect import start_milvusDB_connection, close_milvusDB_connection
import os


NEW_PDF_PATH  = "data"
PDF_ARCHIVE_PATH = "data_archive"
SENTENCE_TRANSFORMER_PATH = "sentence_transformers/sentence-transformers/all-MiniLM-L6-v2"
custom_prompt_template = """Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""

def set_custom_prompt():
    prompt = PromptTemplate(
        template = custom_prompt_template,
        input_variables=['context','question'])
    return prompt


def sentence_transformer_embeddings():
    embeddings = HuggingFaceEmbeddings(
            model_name=SENTENCE_TRANSFORMER_PATH,
            model_kwargs={
                "device": "cuda",
                "trust_remote_code": True 
                },
            encode_kwargs ={
                "normalize_embeddings":False
                }
        )
    return embeddings

# Load language model for response generation
def load_llm():
    accelerator = Accelerator()
    config = {"max_new_tokens": 2048, "context_length":4096,"temperature" : 0.5
              , "gpu_layers":50
              }
    llm = CTransformers(
        model = "model_files/openhermes-2.5-mistral-7b-16k/openhermes-2.5-mistral-7b-16k.Q4_K_M.gguf",
        model_type="mistral",
        config = config
    )
    llm = accelerator.prepare(llm)
    return llm


def retrieval_qa_chain(llm, qa_prompt, db):
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type='stuff',
        retriever=db.as_retriever(search_kwargs={'k': 3}),
        return_source_documents=True,
        chain_type_kwargs={'prompt': qa_prompt}
        ) 
    return qa_chain

# QA chain output model
def qa_bot():
    embeddings = sentence_transformer_embeddings()
    # load vector database
    collection_name = "Document_Reader"
    alias = "Milvusdb" 
    db = start_milvusDB_connection(alias, embeddings, collection_name)
    # load reponse generation model
    llm = load_llm()
    qa_prompt = set_custom_prompt()
    qa = retrieval_qa_chain(llm, qa_prompt, db)
    return qa

#### Streamlit code ####

def conversation_chat(query):
    chain = st.session_state["chain"]
    result = chain.invoke(query)
    print(result)
    print("###")
    print(result['result'])
    st.session_state["history"].append(result['result'])
    return result

def expand_context_container():
    st.session_state["context_expander"] = not st.session_state["context_expander"]
    return

def initialize_session_state():

    if "context_expander" not in st.session_state:
        st.session_state["context_expander"] = True

    if "file_uploader_key" not in st.session_state:
        st.session_state["file_uploader_key"] = 5

    if "history" not in st.session_state:
        st.session_state["history"] = []

    if "source_data" not in st.session_state:
        st.session_state["source_data"] = []

    if "source_doc" not in st.session_state:
        st.session_state["source_doc"] = []

    if "generated" not in st.session_state:
        st.session_state["generated"] = ["Hello! Ask me what you want to find?"]

    if "past" not in st.session_state:
        st.session_state["past"] = ["Hi !!"]

    if "chain" not in st.session_state:
        st.session_state["chain"] = qa_bot()

    st.set_page_config(layout="wide")
    st.session_state["context_container"] , st.session_state["chat_session_container"] = st.columns([1,4] if st.session_state["context_expander"] else [1.5,2.5])
    
    return

def source_update(source_data):
    st.session_state["source_data"]=[]
    st.session_state["source_doc"] = []
    for doc in source_data:
        filename = (doc.metadata['source']).split("/")[-1]
        if filename not in st.session_state["source_doc"]:
            # Get unique source files
            st.session_state["source_doc"].append(filename)
        st.session_state["source_data"].append(f"Document: {doc.metadata['source']}\n Page Number: {doc.metadata['page']}\n\nPage Chunk:\n{doc.page_content}")
    print(st.session_state["source_data"])
    return

        
def display_chat_history():
    response_container = st.container(height=500)
    user_input_container = st.container()

    with user_input_container:
        with st.form(key="abc", clear_on_submit=True):
            user_input = st.text_input("Query: ", placeholder="PDF vector search", key="input_text")
            submit = st.form_submit_button(label="Vector Search")

        if user_input and submit:
            llm_return = conversation_chat(user_input)
            generated_result = llm_return['result']
            source_update(llm_return['source_documents'])
            st.session_state["past"].append(user_input)
            st.session_state["generated"].append(generated_result)

    if st.session_state["generated"]:
        with response_container:
            for i in range(len(st.session_state["generated"])):
                message(
                    st.session_state["past"][i],
                    is_user = True,
                    key = str(i)+ "_user",
                    avatar_style = "thumbs"
                )
                message(
                    st.session_state["generated"][i],
                    key = str(i),
                    avatar_style = "fun-emoji"
                )

def upload_pdf_file():
    st.write("Upload PDF file")
    uploaded_file = st.file_uploader("Choose a PDF file",key=st.session_state["file_uploader_key"])
    if uploaded_file is None:
        st.write("No file uploaded.")
    else:
        # Construct a path to save the file locally. Adjust the path as needed for your environment.
        file_path = f"./{NEW_PDF_PATH}/{uploaded_file.name}"  # This saves the file in the current working directory of your Streamlit app

        # Write the uploaded file to the specified file path
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())  # Use getbuffer() to access the file content

        st.success(f"File '{uploaded_file.name}' saved at '{file_path}'.")

        col1,col2 = st.columns([1,1])     
        with col1:       
            st.button("Injest PDF", on_click=milvusInjest.injest_main, key='full_width_button')
        with col2:
            if st.button("Clear uploaded files"):
                st.session_state["file_uploader_key"] += 1
                st.rerun()


def context_container():
    with st.container():
        st.title("Context Data:")
        if len(st.session_state["source_data"])>0:
            for i in range(len(st.session_state["source_data"])):
                message(
                    st.session_state["source_data"][i],
                    is_user = True,
                    key = str(i)+ "_source",  
                    avatar_style = "fun-emoji"
                )
        else:
            message(
                "No Sources to show",
                is_user = True,
                key = "nosource",
                avatar_style = "fun-emoji"
            )
    return


def download_file(file_name):
    filePath = os.path.join(PDF_ARCHIVE_PATH, file_name)
    with open(filePath, "rb") as pdf_file:
        PDFbyte = pdf_file.read()

    st.download_button(label="Download",
                    data=PDFbyte,
                    file_name=file_name,
                    mime='application/octet-stream')


def directory_explorer():
    filelist=[]
    for _ , dirs, files in os.walk(f"./{PDF_ARCHIVE_PATH}"):
        for filename in st.session_state["source_doc"]:
            filelist.append(filename)
    
    for i in range(len(filelist)):
        st.write(f"{i+1} : {filelist[i]}")
        download_file(filelist[i])



def main():
    initialize_session_state()
    # Left partition Container
    with st.session_state["context_container"]:
        context_expander = st.expander("Context")
        with context_expander:
            st.button("Expand", on_click=expand_context_container)
            context_container()
        with st.container():
            st.title("Download Context Files")
            directory_explorer()


    # Right partition Container
    with st.session_state["chat_session_container"]:
        # Chatbot Container
        with st.container():
            st.title("PDF Reader")
            display_chat_history()    
        # Upload Document Container
        with st.container():
            upload_pdf_file()

if __name__ == "__main__":
    main()
