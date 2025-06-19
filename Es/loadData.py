from langchain_openai import AzureOpenAIEmbeddings

from langchain_elasticsearch import ElasticsearchStore
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import configparser
import os
import time
import numpy as np

print("Starting loading data in ES")

config = configparser.ConfigParser()
config.read('es.env')

os.environ["OPENAI_API_KEY"] = config.get('ES','openai_api_key')
os.environ["AZURE_OPENAI_ENDPOINT"] = config.get('ES','azure_open_ai_endpoint')

#print({section: dict(config[section]) for section in config.sections()})

embeddings=AzureOpenAIEmbeddings(model=config.get('ES','azure_open_ai_emb_model'),
                                 api_version=config.get('ES','azure_open_ai_emb_version'))
text_splitter=RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=0)

upperCaseLetters = ['2', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
for letter in upperCaseLetters:
    loader=DirectoryLoader(path="<PATH>\\rag-graphrag\\full_contract_txt",glob=letter+"*.txt")
    documents = loader.load()
    print("Loading " + str(len(documents)) + " starting with " + letter + "* in ES")
    np_docs = np.array(documents)
    smaller_docs = np.array_split(np_docs, int(len(documents)/int(config.get('ES','batch_size'))))
    for d in smaller_docs:
        docs=text_splitter.split_documents(d.tolist())
        vector_db=ElasticsearchStore.from_documents(
            docs,
            embedding=embeddings,
            index_name=config.get('ES', 'index_name'),
            es_cloud_id=config.get('ES','es_cloud_id'),
            es_api_key=config.get('ES','es_api_key')
        )
        time.sleep(5)
