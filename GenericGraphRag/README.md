# Comparison RAG-ES and GraphRAG-Neo4j

## Setup
- env file:
  - template.env needs to be filled with all the details of your setup (Azure OpenAI details, Neo4j access, paths...)
  - Create your own env files and pass them as an argument when you run the program. Examples:
    - "python create_content.py run.env"
    - "python loadPDFFromContent.py run.env"
- 3 python files:
  - loadDataPDFNoModel.py to generate RAG/GraphRAG without any specific model
  - loadDataPDFWithModel.py to generate RAG/GraphRAG with a specific model filled in a schema.json file
  - chatbot.py to run the streamlit app to query the RAG/GraphRAG
- libs needed:
  - For data loading, *python* and *neo4j_graphrag* (https://pypi.org/project/neo4j-graphrag/, https://neo4j.com/docs/neo4j-graphrag-python/current/)
  - For the chatbot, you need to add streamlit
