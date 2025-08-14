# Comparison RAG-ES and GraphRAG-Neo4j

## Details
- env file:
  - template.env needs to be filled with all the details of your setup (Azure OpenAI details, Neo4j access, paths...)
  - Create your own env files and pass them as an argument when you run the program. Examples:
    - __python create_content.py run.env__
    - __python loadFromContent.py run.env__
    - __streamlit run chatbot.py -- run.env__
- Loading of the GraphRAG, 2 ways:
  - Method 1: extract the content and then load it
    - create_content.py: do a deep extraction of data from your pdf, checking for tables or images. Generates text files to reflect the full content of the PDF
    - loadDataFromContent.py to generate RAG/GraphRAG from the content extracted by the create-content script
  - Method 2: let the graphrag lib do a simple extract
    - loadDataPDF.py to generate RAG/GraphRAG with a specific model filled in a schema.json file
- Using the GraphRAG
  - chatbot.py to run the streamlit app to query the RAG/GraphRAG
## Python libs needed:
  - For the content creation, *python*, *PyMuPDF*, *pdfplumber*, *camelot-py[cv]*, *pytesseract*, *pandas*, *pillow*
  - For data loading, *python* and *neo4j_graphrag* (https://pypi.org/project/neo4j-graphrag/, https://neo4j.com/docs/neo4j-graphrag-python/current/)
  - For the chatbot, *streamlit*
