from langchain.prompts import PromptTemplate
from langchain_community.llms import CTransformers
from langchain.chains.summarize import load_summarize_chain
from pypdf import PdfReader
from langchain.docstore.document import Document
import os
import shutil
import numpy as np
from sklearn.cluster import KMeans
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from accelerate import Accelerator

SENTENCE_TRANSFORMER_PATH = "sentence_transformers/sentence-transformers/all-MiniLM-L6-v2"
PDF_ARCHIVE_PATH = "data_archive"


# Returns summary of list of document(contexts)  
def individual_summaries(llm, selected_docs):
    summary_list = []
    map_prompt = """You will be given a section from a research paper enclosed in triple backticks (```)
    Your goal is to give a summary of the section so that a reader will have a full understanding of what happened in the section.
    The summary should be 2-3 sentences.

    ```{text}```
    SUMMARY:
    """
    map_prompt_template = PromptTemplate(template=map_prompt, input_variables=["text"])

    prompt = PromptTemplate(
        template = map_prompt,
        input_variables=['text'])

    summary_chain = load_summarize_chain(llm = llm, chain_type="stuff", prompt = prompt)

    #For list of docs
    for i, doc in enumerate(selected_docs):
        # Summary of Context chunk
        chunk_summary = summary_chain.invoke([doc])
        # Chunk Summary added to list
        summary_list.append(chunk_summary['output_text'])
        print(chunk_summary['output_text'])
    
    summaries = "\n".join(summary_list)
    

    return summaries


# Summary of all chunks 
def generate_summary(llm, abstract_summary):
    summary_prompt_template = """You will be given a series of sentences from a research paper.
    The sentences will be enclosed in triple backticks (```)
    Your response should be in bullet points and fully encompass what was said in the sentences.

    ```{text}```

    VERBOSE SUMMARY:
    """

    prompt = PromptTemplate(
        template = summary_prompt_template,
        input_variables=['text'])

    summary_chain = load_summarize_chain(llm = llm, chain_type="stuff", prompt = prompt)

    summary =summary_chain.invoke([abstract_summary])
    return summary


# Main Summary Function
def create_abstract_summmaries(file_name):
    accelerator = Accelerator()
    llm = llm_load()
    embeddings = create_embeddings()
    llm, embeddings = accelerator.prepare(llm, embeddings)

    data_path =  PDF_ARCHIVE_PATH
    summary_list = []

    file_path = os.path.join(data_path, file_name)
    reader = PdfReader(file_path)
        
    page_text=""
    for i in range(len(reader.pages)):
        page_text = page_text +"\n"+ reader.pages[i].extract_text()
            
    text_splitter = RecursiveCharacterTextSplitter( chunk_size=2000, chunk_overlap=200)
    docs = text_splitter.create_documents([page_text])

    num_clusters = 11
    vectors = embeddings.embed_documents([x.page_content for x in docs])
    print("docs length:",len(docs))
    print("vector length:",len(vectors))
    print("kmeans:\n")
    kmeans = KMeans(n_clusters=num_clusters, random_state=42).fit(vectors)
    print(kmeans)
    closest_indices = []

    # Loop through the number of clusters you have
    for i in range(num_clusters):
        # Get the list of distances from that particular cluster center
        distances = np.linalg.norm(vectors - kmeans.cluster_centers_[i], axis=1)
        # Find the list position of the closest one (using argmin to find the smallest distance)
        closest_index = np.argmin(distances)
        # Append that position to your closest indices list
        closest_indices.append(closest_index)

    selected_indices = sorted(closest_indices)
        
    selected_docs = [docs[doc] for doc in selected_indices]

    summaries = individual_summaries(llm, selected_docs)

    summaries = Document(page_content=summaries)

    final_summary = generate_summary(llm, summaries)
    print("\n## Final Summary Generated ##")

    return final_summary




def create_embeddings():
    model_path = SENTENCE_TRANSFORMER_PATH
    embeddings = HuggingFaceEmbeddings(
        model_name=model_path,
        model_kwargs={
            "device": "cuda",
            "trust_remote_code": True
            },
        encode_kwargs ={
            "normalize_embeddings":False
            }
    )
    return embeddings

def llm_load():
    config = { "max_new_tokens": 2096, "context_length":4096,"temperature" : 0.7, "gpu_layers":20 }
    llm = CTransformers(
        model = "model_files/openhermes-2.5-mistral-7b-16k/openhermes-2.5-mistral-7b-16k.Q4_K_M.gguf",
        model_type="mistral",
        config = config
    )
    return llm

summary = create_abstract_summmaries('2302.13971.pdf')
print(summary["output_text"])

