# Comparison RAG-ES and GraphRAG-Neo4j

## Setup
- env file:
  - template.env needs to be filled with all the details of your setup (Azure OpenAI details, Neo4j access...)
- 3 python files:
  - loadDataPDFNoModel.py to generate RAG/GraphRAG without any specific model
  - loadDataPDFWithModel.py to generate RAG/GraphRAG with a specific model filled in a schema.json file
  - chatbot.py to run the streamlit app to query the RAG/GraphRAG
- libs needed:
  - neo4j_graphrag (https://pypi.org/project/neo4j-graphrag/, https://neo4j.com/docs/neo4j-graphrag-python/current/)
  - streamlit
