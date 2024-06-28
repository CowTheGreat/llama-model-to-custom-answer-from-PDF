# llama-model-to-custom-answer-from-PDF
Easy Local Rag is a user-friendly system that converts PDFs into searchable text. Using advanced NLP and machine learning, users can upload PDFs, extract relevant content, and get answers to specific queries.

Setup
1. git clone https://github.com/AllAboutAI-YT/easy-local-rag.git
2. cd dir
3. pip install -r requirements.txt
4. Install Ollama (https://ollama.com/download)
5. ollama pull llama3 (etc)
6. ollama pull mxbai-embed-large
7. use streamlit run fianl.py to automatically start the srever and directly use the GUI

Referred : https://www.youtube.com/watch?v=Oe-7dGDyzPM

Additional ways
1. run appli.py for tkinter u might need to manually start the ollama server next time using ollama pull llama3
2. 2. streamlit run server.py to use streamlit interface where you might need to manually start the ollama server next time using ollama pull llama3
