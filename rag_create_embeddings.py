import os
import chromadb
from langchain.vectorstores.chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import DirectoryLoader
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
import requests
import base64
from google.cloud import secretmanager


import traceback
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.prompts import PromptTemplate
#!pip install pypdf, langchain, chromadb, sentence-transformers
#!pip install --upgrade --quiet  langchain-google-genai pillow
import shutil
from google.cloud import secretmanager

loader = DirectoryLoader('./articles_rep/', glob="./*.pdf", loader_cls=PyPDFLoader)

documents = loader.load()

#splitting the text into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
texts = text_splitter.split_documents(documents)
print("Number of Chunks Created:", len(texts))

chroma_dir = './chroma_db'
shutil.rmtree(chroma_dir, ignore_errors=True)
#Store in vector database
client = chromadb.PersistentClient(path=chroma_dir)
embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
vectordb = Chroma(client=client, embedding_function=embedding_function, collection_name="articles_embeddings")
vectordb_ids = vectordb.add_documents(texts)
print(f"Added {len(vectordb_ids)} chunks to the Chroma database")
#zipping Chroma Db
shutil.make_archive('chroma_db', 'zip', chroma_dir)
chroma_dir = './chroma_db'
shutil.rmtree(chroma_dir, ignore_errors=True)
shutil.unpack_archive("chroma_db.zip", chroma_dir)
filter_list = []
#Make a retrieval object
client = chromadb.PersistentClient(path=chroma_dir)
embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
vectordb = Chroma(client=client, embedding_function=embedding_function, collection_name="articles_embeddings")
doc_list = list(set( [ meta['source'] for meta  in vectordb.get()['metadatas'] ]))
def llm_output(llm_response):
    print("Answer from LLM:",llm_response['answer'])
    print('\n\nSources:')
    for source in llm_response["context"]:
         print(source.metadata['source'], ' Page:', source.metadata['page'] )


from google.cloud import secretmanager
from google.api_core.exceptions import NotFound


llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=GOOGLE_API_KEY)
print(llm.invoke("tell me a joke").content)


         
