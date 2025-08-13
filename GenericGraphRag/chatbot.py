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
st.html("<h1 style='font-size:35px;'><a href='https://neo4j.com/generativeai/'>Neo4j GraphRAG</a></h1>")

if 'config' not in st.session_state:
    st.session_state.config = configparser.ConfigParser()
    st.session_state.config.read('run1.env') # TO BE CHANGED TO YOUR FILE

#### Program variable setup
# Neo4j
URI = st.session_state.config.get('Conf','neo4j_uri')
PASSWORD = st.session_state.config.get('Conf','neo4j_password')
USER = st.session_state.config.get('Conf','neo4j_user')
DATABASE = st.session_state.config.get('Conf','neo4j_database')
# OPENAI
END_POINT = st.session_state.config.get('Conf','azure_open_ai_endpoint')
OPENAI_KEY = st.session_state.config.get('Conf','openai_api_key')
# LLM
MODEL_NAME = st.session_state.config.get('Conf','azure_open_ai_model')
API_VERSION = st.session_state.config.get('Conf','openai_api_version')
# FILES
FILES_PATH = st.session_state.config.get('Conf','files_path') # "C:\\Users\\XavierPilas\\Documents\\GitHub\\work-rag-graphrag\\10k\\"
# EMBEDDINGS
EMB_MODEL_NAME = st.session_state.config.get('Conf','azure_open_ai_emb_model')
EMB_API_VERSION = st.session_state.config.get('Conf','azure_open_ai_emb_version')

#Environment variables setup
os.environ["AZURE_OPENAI_ENDPOINT"] = END_POINT
os.environ["OPENAI_API_KEY"] = OPENAI_KEY
os.environ["AZURE_OPENAI_API_KEY"] = OPENAI_KEY

if 'neo4j_driver' not in st.session_state:
    st.session_state.neo4j_driver = neo4j.GraphDatabase.driver(URI, auth=(USER, PASSWORD), database=DATABASE)

if 'llm' not in st.session_state:
    st.session_state.llm = AzureOpenAILLM(model_name=MODEL_NAME,
                                          api_version=API_VERSION,
                                          model_params={"temperature": 0.0})

if 'embeddings' not in st.session_state:
    st.session_state.embeddings = AzureOpenAIEmbeddings(api_version=API_VERSION)

if 'rag_template' not in st.session_state:
    st.session_state.rag_template = RagTemplate(template='''
Answer the Question using the following Context.
Only respond with information mentioned in the Context.
Do not inject any speculative information not mentioned. 

# Question:
{query_text}

# Context:
{context}

# Answer:
''', expected_inputs=['query_text', 'context'])

if 'vc_retriever' not in st.session_state:
    st.session_state.vc_retriever = VectorCypherRetriever(
        st.session_state.neo4j_driver,
        index_name="text_embeddings",
        embedder=st.session_state.embeddings,
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

if 'vc_rag' not in st.session_state:
    st.session_state.vc_rag = GraphRAG(llm=st.session_state.llm, retriever=st.session_state.vc_retriever, prompt_template=st.session_state.rag_template)

q = st.text_area("Enter your question")
if q is not None and len(q) > 0:
    btn=st.button("submit")
    if btn:
        vc_rag_result = st.session_state.vc_rag.search(q, retriever_config={'top_k': 5}, return_context=True)
        st.write(f"GraphRAG Response: \n{vc_rag_result.answer}")

        ### SOURCES CHUNKS
        last = vc_rag_result.retriever_result.items[0].content.find('=== kg_rels ===')
        text_chunks = vc_rag_result.retriever_result.items[0].content[28:last]
        chunks = text_chunks.split('-$$-')
        st.divider()
        st.write("**Sources chunks:**")
        for cc in chunks:
            if cc[0] == 'P':
                c = cc
            else:
                c = cc[2:]
            pos = c.index('=')
            pageNum = c[:c.index('=')].strip()
            st.badge("Chunk on " + pageNum)
            st.write(c[:200].replace(pageNum, '').replace('=', '').replace('MAIN TEXT: ', '').strip() + " ...")
        st.divider()
        #### NODES and RELS
        text_nodes_rels = vc_rag_result.retriever_result.items[0].content[last+15:]
        nodes_rels = text_nodes_rels.split('-$$-')
        nodes = set()
        rels = set()
        for nr in nodes_rels[:50]:
            if len(nr[2:-2]) > 5:
                infos = nr[2:-2].replace(' - ', ';').replace(' -> ', ';').split(';')
                nodes.add(infos[0])
                rels.add(infos[1])
                nodes.add(infos[2])
        st.badge("Nodes used")
        st.write(", ".join(list(nodes)))
        st.badge("Relationships used")
        st.write(", ".join(list(rels)))