# Comparison RAG-ES and GraphRAG-Neo4j

## Setup
- env file:
  - template.env needs to be filled with all the details of your setup (Azure OpenAI details, Neo4j access...)
- 2 python files:
  - loadData.py to generate RAG/GraphRAG
  - chatbot.py to run the streamlit app to query the RAG/GraphRAG
- libs needed:
  - neo4j_graphrag (https://pypi.org/project/neo4j-graphrag/, https://neo4j.com/docs/neo4j-graphrag-python/current/)
  - streamlit
