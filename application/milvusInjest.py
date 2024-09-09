from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Milvus
import os
import shutil


def check_pdf_files(NEW_PDF_PATH, PDF_ARCHIVE_PATH):
    new_pdf_flag = False
    if os.path.exists(NEW_PDF_PATH):
        print("NEW_PDF_PATH path exists")
    else:
        print("NEW_PDF_PATH path does not exist")
        return new_pdf_flag
    
    if os.path.exists(PDF_ARCHIVE_PATH):
        print("PDF_ARCHIVE_PATH path exists")
    else:
        os.makedirs(PDF_ARCHIVE_PATH)
        print("NEW_PDF_PATH path created")
    
    new_pdf_files = os.listdir(NEW_PDF_PATH)
    pdf_archive_files = os.listdir(PDF_ARCHIVE_PATH)
    
    for file_name in new_pdf_files:
        if file_name.endswith('.pdf') and file_name in pdf_archive_files:
            file_path = os.path.join(NEW_PDF_PATH, file_name)
            os.remove(file_path)
            print(f"Deleted {file_name} as same filename exists in Archives.")
        elif file_name.endswith('.pdf'):
            new_pdf_flag = True

    return new_pdf_flag


def move_pdfs_to_archive(NEW_PDF_PATH, PDF_ARCHIVE_PATH):
    for file_name in os.listdir(NEW_PDF_PATH):
        if file_name.endswith('.pdf'):
            source_file = os.path.join(NEW_PDF_PATH, file_name)
            if os.path.isfile(source_file):
                destination_file = os.path.join(PDF_ARCHIVE_PATH, file_name)
                shutil.move(source_file, destination_file)
                print(f"Moved: {file_name}")


def create_vectors(SENTENCE_TRANSFORMER_PATH, NEW_PDF_PATH, collection_name):
    model_path = SENTENCE_TRANSFORMER_PATH
    data_path =  NEW_PDF_PATH
    loader = DirectoryLoader(data_path, glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()

    # There are other text splitters too. But below is widely used.
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    docs = splitter.split_documents(documents)

    # It uses local models if tokenizer_config present, else if only model_config, it uses default pooling.
    # It also downloads the model locally from HuggingFace repo if available there
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

    # Creating vector_store
    print("milvus")
    vectorstore = Milvus.from_documents(
        docs ,
        embeddings,
        connection_args={"host": "127.0.0.1", "port": "19530"},
        collection_name = collection_name, ## custom collection name 
        search_params = {"metric":"IP","offset":0} ## search params
    )
    return vectorstore
    # Vectorizes the text based on the data the embedding model was trained on
    # vectorstore.save_local(DB_Milvus_PATH)


def injest_main():
    
    NEW_PDF_PATH  = "data"
    SENTENCE_TRANSFORMER_PATH = "sentence_transformers/sentence-transformers/all-MiniLM-L6-v2"
    PDF_ARCHIVE_PATH = "data_archive"
    collection_name = "Document_Reader"
    
    new_pdf_flag = check_pdf_files(NEW_PDF_PATH, PDF_ARCHIVE_PATH)
    if new_pdf_flag==True:
        msg ="New pdf files detected\nAdding to Vector Database"
        create_vectors(SENTENCE_TRANSFORMER_PATH, NEW_PDF_PATH, collection_name)
        move_pdfs_to_archive(NEW_PDF_PATH, PDF_ARCHIVE_PATH)
    else:
        msg = "No New pdf files detected"
    print(msg)
    return msg

# if __name__ == '__main__':
#     injest_main()
