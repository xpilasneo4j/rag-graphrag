from neo4j_graphrag.retrievers import VectorRetriever
import neo4j
from neo4j_graphrag.llm import AzureOpenAILLM
from neo4j_graphrag.embeddings.openai import AzureOpenAIEmbeddings
import streamlit as st
from neo4j_graphrag.retrievers import VectorCypherRetriever
from neo4j_graphrag.generation import RagTemplate
from neo4j_graphrag.generation.graphrag import GraphRAG
import configparser
import os

config = configparser.ConfigParser()
config.read('neo4j.env')

os.environ["AZURE_OPENAI_ENDPOINT"] = config.get('Neo4j','azure_open_ai_endpoint')
os.environ["OPENAI_API_KEY"] = config.get('Neo4j','openai_api_key')
os.environ["AZURE_OPENAI_API_KEY"] = config.get('Neo4j','openai_api_key')

# Neo4j db infos
URI = config.get('Neo4j','neo4j_uri')
AUTH = ("neo4j", config.get('Neo4j','neo4j_password'))
DATABASE = config.get('Neo4j','neo4j_database')

st.title("NEO4J GRAPHRAG EXAMPLE")

rag_template = RagTemplate(template='''Answer the Question using the following Context. Only respond with information mentioned in the Context. Do not inject any speculative information not mentioned. 

# Question:
{query_text}

# Context:
{context}

# Answer:
''', expected_inputs=['query_text', 'context'])

llm = AzureOpenAILLM(model_name=config.get('Neo4j','azure_open_ai_model'),
                     api_version=config.get('Neo4j','openai_api_version'),
                     model_params={"temperature": 0.0})


embedder = AzureOpenAIEmbeddings(api_version=config.get('Neo4j','openai_api_version'))

driver = neo4j.GraphDatabase.driver(URI, auth=AUTH, database=DATABASE)

vector_retriever = VectorRetriever(
    driver,
    index_name="text_embeddings",
    embedder=embedder,
    return_properties=["text"],
)

vc_retriever = VectorCypherRetriever(
    driver,
    index_name="text_embeddings",
    embedder=embedder,
    retrieval_query="""
// 1) Go out 2-3 hops in the entity graph and get relationships
WITH node AS chunk
MATCH (chunk)<-[:FROM_CHUNK]-()-[relList:!FROM_CHUNK]-{1,3}()
UNWIND relList AS rel

// 2) Collect relationships and text chunks
WITH collect(DISTINCT chunk) AS chunks, 
  collect(DISTINCT rel) AS rels

// 3) Format and return context
RETURN '=== text ===\n' + apoc.text.join([c in chunks | c.text], '\n---\n') + '\n\n=== kg_rels ===\n' +
  apoc.text.join([r in rels | startNode(r).name + ' - ' + type(r)  +  ' -> ' + endNode(r).name ], '\n---\n') AS info
"""
)
v_rag  = GraphRAG(llm=llm, retriever=vector_retriever, prompt_template=rag_template)
vc_rag = GraphRAG(llm=llm, retriever=vc_retriever, prompt_template=rag_template)

q = st.text_area("enter your question")
if q is not None:
    btn=st.button("submit")
    if btn:
        v_rag_result = v_rag.search(q, retriever_config={'top_k': 5}, return_context=True)
        vc_rag_result = vc_rag.search(q, retriever_config={'top_k': 5}, return_context=True)

        st.write(f"Vector Response: \n{v_rag_result.answer}")
        st.write("\n===========================\n")
        st.write(f"Vector + Cypher Response: \n{vc_rag_result.answer}")


                    