import asyncio
import logging
from langchain_community.document_loaders import DirectoryLoader

import neo4j
from neo4j_graphrag.embeddings import AzureOpenAIEmbeddings
from neo4j_graphrag.experimental.pipeline.kg_builder import SimpleKGPipeline
from neo4j_graphrag.experimental.pipeline.pipeline import PipelineResult
from neo4j_graphrag.experimental.pipeline.types.schema import (
    EntityInputType,
    RelationInputType,
)
from neo4j_graphrag.indexes import create_vector_index
from neo4j_graphrag.llm import LLMInterface
from neo4j_graphrag.llm import AzureOpenAILLM
import configparser
import os
import time

config = configparser.ConfigParser()
config.read('neo4j.env')

logging.basicConfig()
#logging.getLogger("neo4j_graphrag").setLevel(logging.DEBUG)
logging.getLogger("neo4j_graphrag").setLevel(logging.INFO)

os.environ["AZURE_OPENAI_ENDPOINT"] = config.get('Neo4j','azure_open_ai_endpoint')
os.environ["OPENAI_API_KEY"] = config.get('Neo4j','openai_api_key')
os.environ["AZURE_OPENAI_API_KEY"] = config.get('Neo4j','openai_api_key')

# Neo4j db infos
URI = config.get('Neo4j','neo4j_uri')
AUTH = ("neo4j", config.get('Neo4j','neo4j_password'))
DATABASE = config.get('Neo4j','neo4j_database')

driver = neo4j.GraphDatabase.driver(URI, auth=AUTH, database=DATABASE)
create_vector_index(driver, name="text_embeddings", label="Chunk",
                    embedding_property="embedding", dimensions=1536, similarity_fn="cosine")
del driver

# Instantiate Entity and Relation objects. This defines the
# entities and relations the LLM will be looking for in the text.

ENTITIES: list[EntityInputType] = [
    # Core Business Entities
    "Person",
    {"label": "Company", "description": "Business entity, corporation, or organization"},
    {"label": "Legal_Entity", "description": "Subsidiary, affiliate, or related business entity"},
    
    # Agreement and Legal Structures
    {"label": "Agreement", "description": "Contract, agreement, or legal document", 
     "properties": [{"name": "agreement_type", "type": "STRING"}, 
                   {"name": "effective_date", "type": "DATE"},
                   {"name": "termination_date", "type": "DATE"}]},
    
    # Products and Services
    {"label": "Product", "description": "Physical products, goods, or merchandise"},
    {"label": "Service", "description": "Services, consulting, or intangible offerings"},
    {"label": "Technology", "description": "Software, systems, or technical solutions"},
    
    # Financial and Legal Terms
    {"label": "Financial_Terms", "description": "Payment terms, fees, compensation structures",
     "properties": [{"name": "amount", "type": "STRING"}, 
                   {"name": "currency", "type": "STRING"},
                   {"name": "payment_schedule", "type": "STRING"}]},
    
    {"label": "Intellectual_Property", "description": "Patents, trademarks, copyrights, trade secrets"},
    
    # Geographic and Temporal
    {"label": "Location", "description": "Business addresses, territories, jurisdictions"},
    {"label": "Territory", "description": "Geographic regions or markets covered"},
    {"label": "Time_Period", "description": "Contract terms, deadlines, milestones"},
    
    # Specialized Entities
    {"label": "Regulatory_Body", "description": "FDA, IRB, or other regulatory authorities"},
    {"label": "Study", "description": "Clinical trials, research studies, or investigations"}
]

RELATIONS: list[RelationInputType] = [
    # Core Agreement Relationships
    "PARTY_TO",
    {"label": "PROVIDES", "description": "Entity providing products or services to another"},
    {"label": "RECEIVES", "description": "Entity receiving products or services"},
    
    # Financial Relationships
    {"label": "PAYS", "description": "Payment relationship between entities",
     "properties": [{"name": "amount", "type": "STRING"},
                   {"name": "frequency", "type": "STRING"}]},
    
    {"label": "LICENSES", "description": "Licensing relationship for IP or products"},
    {"label": "DISTRIBUTES", "description": "Distribution or resale relationship"},
    
    # Operational Relationships
    {"label": "CONDUCTS", "description": "Entity conducting studies, research, or operations"},
    {"label": "SPONSORS", "description": "Sponsorship or funding relationship"},
    {"label": "ENDORSES", "description": "Endorsement or promotional relationship"},
    {"label": "MANUFACTURES", "description": "Manufacturing or production relationship"},
    
    # Legal and Compliance
    {"label": "INDEMNIFIES", "description": "Indemnification obligation between parties"},
    {"label": "GOVERNED_BY", "description": "Legal jurisdiction or governing law"},
    {"label": "OWNS", "description": "Ownership of assets, IP, or entities"},
    {"label": "LICENSED_BY", "description": "Entity licensed or regulated by another entity"},
    
    # Geographic and Temporal
    {"label": "LOCATED_AT", "description": "Physical location or business address"},
    {"label": "OPERATES_IN", "description": "Business operations in specific territory"},
    {"label": "EFFECTIVE_DURING", "description": "Time period when relationship is active"},
    
    # Specialized Relationships
    {"label": "AFFILIATED_WITH", "description": "Corporate affiliation or subsidiary relationship"},
    {"label": "COMPETES_WITH", "description": "Competitive relationship or exclusivity"},
    {"label": "COLLABORATES_WITH", "description": "Joint venture or strategic partnership"},
    
    # Termination and Changes
    {"label": "TERMINATES", "description": "Termination conditions or events",
     "properties": [{"name": "termination_type", "type": "STRING"},
                   {"name": "notice_period", "type": "STRING"}]}
]

POTENTIAL_SCHEMA = [
    # Core Agreement Structure
    ("Company", "PARTY_TO", "Agreement"),
    ("Person", "PARTY_TO", "Agreement"),
    ("Agreement", "GOVERNED_BY", "Location"),
    
    # Business Operations
    ("Company", "PROVIDES", "Product"),
    ("Company", "PROVIDES", "Service"),
    ("Company", "LICENSES", "Technology"),
    ("Company", "DISTRIBUTES", "Product"),
    ("Company", "MANUFACTURES", "Product"),
    
    # Financial Relationships
    ("Company", "PAYS", "Financial_Terms"),
    ("Person", "RECEIVES", "Financial_Terms"),
    ("Company", "SPONSORS", "Study"),
    
    # Geographic and Legal
    ("Company", "LOCATED_AT", "Location"),
    ("Company", "OPERATES_IN", "Territory"),
    ("Agreement", "EFFECTIVE_DURING", "Time_Period"),
    
    # Intellectual Property
    ("Company", "OWNS", "Intellectual_Property"),
    ("Company", "LICENSES", "Intellectual_Property"),
    
    # Specialized Business Relationships
    ("Company", "ENDORSES", "Product"),
    ("Company", "COLLABORATES_WITH", "Company"),
    ("Company", "AFFILIATED_WITH", "Legal_Entity"),
    ("Company", "CONDUCTS", "Study"),
    
    # Legal Obligations
    ("Company", "INDEMNIFIES", "Company"),
    ("Agreement", "TERMINATES", "Time_Period"),
    
    # Regulatory
    ("Study", "GOVERNED_BY", "Regulatory_Body"),
    ("Company", "LICENSED_BY", "Regulatory_Body")
]

embeddings=AzureOpenAIEmbeddings(model=config.get('Neo4j','azure_open_ai_emb_model'),
                                 api_version=config.get('Neo4j','azure_open_ai_emb_version'))

async def define_and_run_pipeline(
    neo4j_driver: neo4j.Driver,
    llm: LLMInterface,
    content: str
) -> PipelineResult:
    # Create an instance of the SimpleKGPipeline
    kg_builder = SimpleKGPipeline(
        llm=llm,
        driver=neo4j_driver,
        embedder=embeddings,
        entities=ENTITIES,
        relations=RELATIONS,
        potential_schema=POTENTIAL_SCHEMA,
        from_pdf=False,
        neo4j_database=DATABASE,
    )
    return await kg_builder.run_async(text=content)


async def main(content: str) -> PipelineResult:
    llm = AzureOpenAILLM(
        model_name=config.get('Neo4j','azure_open_ai_model'),
        api_version=config.get('Neo4j','openai_api_version'),
        model_params={
            "max_tokens": 5000,
            "response_format": {"type": "json_object"},
        },
    )
    with neo4j.GraphDatabase.driver(URI, auth=AUTH, database=DATABASE) as driver:
        res = await define_and_run_pipeline(driver, llm, content)
    await llm.async_client.close()
    return res

if __name__ == "__main__":
    # use 10_contacts_txt for testing before loading all
    loader=DirectoryLoader(path="<PATH>\\rag-graphrag\\full_contract_txt",glob="*.txt")
    documents=loader.load()
    i = 1
    for d in documents:
        print("Load document " + str(i) + "/" + str(len(documents)))
        res = asyncio.run(main(d.page_content))
        print(res)
        time.sleep(5)
        i = i + 1
