import asyncio
import configparser
import os
import time
import neo4j
from neo4j_graphrag.embeddings import AzureOpenAIEmbeddings
from neo4j_graphrag.experimental.pipeline.kg_builder import SimpleKGPipeline
from neo4j_graphrag.experimental.pipeline.pipeline import PipelineResult
from neo4j_graphrag.llm import LLMInterface
from neo4j_graphrag.llm import AzureOpenAILLM
from neo4j_graphrag.indexes import create_vector_index

# Config loading
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
EMB_SIZE = config.get('Conf','emb_size')

#Environment variables setup
os.environ["AZURE_OPENAI_ENDPOINT"] = END_POINT
os.environ["OPENAI_API_KEY"] = OPENAI_KEY
os.environ["AZURE_OPENAI_API_KEY"] = OPENAI_KEY

# Building Neo4j and Embeddings components
NEO4J_DRIVER = neo4j.GraphDatabase.driver(URI, auth=(USER, PASSWORD), database=DATABASE)
create_vector_index(NEO4J_DRIVER, name="text_embeddings", label="Chunk",
                    embedding_property="embedding", dimensions=int(EMB_SIZE), similarity_fn="cosine")

EMBEDDINGS=AzureOpenAIEmbeddings(
    model=EMB_MODEL_NAME,
    api_version=EMB_API_VERSION
)

# Pipeline execution
async def define_and_run_pipeline(
        llm: LLMInterface,
        file_path: str
) -> PipelineResult:
    kg_builder = SimpleKGPipeline(
        llm=llm,
        driver=NEO4J_DRIVER,
        embedder=EMBEDDINGS,
        from_pdf=True,
        neo4j_database=DATABASE,
    )
    return await kg_builder.run_async(file_path=file_path)

# Main for one file
async def main(file_path: str) -> PipelineResult:
    llm = AzureOpenAILLM(
        model_name=MODEL_NAME,
        api_version=API_VERSION,
        model_params={
            "max_tokens": 5000,
            "response_format": {"type": "json_object"},
        }
    )
    res_pipeline = await define_and_run_pipeline(llm, file_path)
    await llm.async_client.close()
    return res_pipeline

### MAIN
if __name__ == "__main__":
    # Iterating for each PDF of the directory
    for name in os.listdir(FILES_PATH):
        if name.endswith(".pdf"):
            print(" ----------- Start " + name)
            print(asyncio.run(main(FILES_PATH + name)))
            time.sleep(5) # Needed for LLM Token Limit
            print(" ----------- END " + name)
