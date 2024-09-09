from pymilvus import connections
from langchain_community.vectorstores import Milvus
from langchain_community.embeddings import HuggingFaceEmbeddings



def start_milvusDB_connection(alias, embeddings, collection_name):
    connections.connect(
        alias=alias,  # You can use an alias for the connection
        host="127.0.0.1",  # Replace with the IP address or hostname of the Milvus server
        port="19530"  # Replace with the correct port if not using the default
    )
    vectordb = Milvus.from_documents(
        {},
        embeddings,
        collection_name = collection_name,
        connection_args={"host": "127.0.0.1", "port":"19530"},
        consistency_level="Strong"
        )
    return vectordb

def close_milvusDB_connection(alias):
    connections.disconnect(
        alias=alias,  # You can use an alias for the connection
    )

