import configparser
import os
import streamlit as st
import neo4j
from neo4j_graphrag.retrievers import VectorRetriever
from neo4j_graphrag.llm import AzureOpenAILLM
from neo4j_graphrag.embeddings.openai import AzureOpenAIEmbeddings
from neo4j_graphrag.retrievers import VectorCypherRetriever
from neo4j_graphrag.generation import RagTemplate
from neo4j_graphrag.generation.graphrag import GraphRAG

st.set_page_config(layout="wide")
st.title("NEO4J GRAPHRAG CHATBOT")
st.html("<h1 style='font-size:15px;'><a href='https://neo4j.com/generativeai/'>Neo4j GraphRAG</a></h1>")

config = configparser.ConfigParser()
config.read('run1.env') # TO BE CHANGED TO YOUR FILE

#### Program variable setup
# Neo4j
URI = config.get('Conf','neo4j_uri')
PASSWORD = config.get('Conf','neo4j_password')
USER = config.get('Conf','neo4j_user')
DATABASE = config.get('Conf','neo4j_database')
# OPENAI
END_POINT = config.get('Conf','azure_open_ai_endpoint')
OPENAI_KEY = config.get('Conf','openai_api_key')
# LLM
MODEL_NAME = config.get('Conf','azure_open_ai_model')
API_VERSION = config.get('Conf','openai_api_version')
# FILES
FILES_PATH = config.get('Conf','files_path')
# EMBEDDINGS
EMB_MODEL_NAME = config.get('Conf','azure_open_ai_emb_model')
EMB_API_VERSION = config.get('Conf','azure_open_ai_emb_version')

#Environment variables setup
os.environ["AZURE_OPENAI_ENDPOINT"] = END_POINT
os.environ["OPENAI_API_KEY"] = OPENAI_KEY
os.environ["AZURE_OPENAI_API_KEY"] = OPENAI_KEY

neo4j_driver = neo4j.GraphDatabase.driver(URI, auth=(USER, PASSWORD), database=DATABASE)

llm = AzureOpenAILLM(model_name=MODEL_NAME,
                     api_version=API_VERSION,
                     model_params={"temperature": 0.0})

embeddings = AzureOpenAIEmbeddings(api_version=API_VERSION)

rag_template = RagTemplate(template='''
Answer the Question using the following Context.
Only respond with information mentioned in the Context.
Do not inject any speculative information not mentioned. 

# Question:
{query_text}

# Context:
{context}

# Answer:
''', expected_inputs=['query_text', 'context'])

vc_retriever = VectorCypherRetriever(
    neo4j_driver,
    index_name="text_embeddings",
    embedder=embeddings,
    retrieval_query="""
// 1) Go out 2-3 hops in the entity graph and get relationships
WITH node AS chunk
MATCH (chunk)<-[:FROM_CHUNK]-()-[relList:!FROM_CHUNK]-{1,3}()
UNWIND relList AS rel

// 2) Collect relationships and text chunks
WITH collect(DISTINCT chunk) AS chunks, 
  collect(DISTINCT rel) AS rels

// 3) Format and return context
RETURN '=== text === ' + apoc.text.join([c in chunks | c.text], ' -$$- ') + ' === kg_rels === ' +
  apoc.text.join([r in rels | startNode(r).name + ' - ' + type(r)  +  ' -> ' + endNode(r).name ], ' --- ') AS info
"""
)

vc_rag = GraphRAG(llm=llm, retriever=vc_retriever, prompt_template=rag_template)

q = st.text_area("Enter your question")
if q is not None and len(q) > 0:
    btn=st.button("submit")
    if btn:
        vc_rag_result = vc_rag.search(q, retriever_config={'top_k': 5}, return_context=True)
        st.write(f"GraphRAG Response: \n{vc_rag_result.answer}")

        ### SOURCES CHUNKS
        last = vc_rag_result.retriever_result.items[0].content.find(' === kg_rels === ')
        text_chunks = vc_rag_result.retriever_result.items[0].content[27:last]
        chunks = text_chunks.split(' -$$- ')
        st.divider()
        st.write("**Sources chunks:**")
        cpt = 1
        for cc in chunks:
            st.badge("Chunk #" + str(cpt))
            st.write(cc.strip()[:2000] + " ...")
            cpt = cpt + 1
        st.divider()
        #### NODES and RELS
        text_nodes_rels = vc_rag_result.retriever_result.items[0].content[last+15:]
        nodes_rels = text_nodes_rels.split(' --- ')
        nodes = set()
        rels = set()
        for nrs in nodes_rels:
            nr = nrs.strip()
            if nr != 'null' and len(nr) > 5:
                infos = nr.replace(' - ', ';').replace(' -> ', ';').split(';')
                if len(infos) == 3:
                    nodes.add(infos[0])
                    rels.add(infos[1])
                    nodes.add(infos[2])
        st.badge("Nodes used")
        st.write(", ".join(list(nodes)))
        st.badge("Relationships used")
        st.write(", ".join(list(rels)))