from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain_elasticsearch import ElasticsearchStore
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
import streamlit as st
import configparser
import os

config = configparser.ConfigParser()
config.read('es.env')

os.environ["OPENAI_API_KEY"] = config.get('ES','openai_api_key')
os.environ["AZURE_OPENAI_ENDPOINT"] = config.get('ES','azure_open_ai_endpoint')

template="""
You are an assistant for question-answering tasks. 
Use the following pieces of retrieved context to answer the question. 
If you don't know the answer, just say that you don't know. 
Use five sentences minimum and keep the answer concise.
Question: {question} 
Context: {context} 
Answer:
"""
embeddings = AzureOpenAIEmbeddings(api_version=config.get('ES','openai_api_version'))

vector_db=ElasticsearchStore(
    embedding=embeddings,
    index_name=config.get('ES', 'index_name'),
    es_cloud_id=config.get('ES','es_cloud_id'),
    es_api_key=config.get('ES','es_api_key')
)
retriever=vector_db.as_retriever()
prompt = ChatPromptTemplate.from_template(template)
llm = AzureChatOpenAI(azure_deployment=config.get('41','azure_open_ai_model'),
                      api_version=config.get('41','openai_api_version'),
                      temperature=0)
st.title("ELASTIC SEARCH RAG")
question=st.text_area("enter your question")
if question is not None:
    btn=st.button("submit")
    if btn:
        rag_chain=(
            {"context":retriever,"question":RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        response=rag_chain.invoke(question)
        st.write(response)

