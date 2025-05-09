# Django RAG Chatbot

A Django-based Retrieval-Augmented Generation (RAG) chatbot application that uses FAISS for vector indexing, Mistral-7B for text generation, and MySQL as the database. The application allows users to upload text documents, add text via a textarea, and query the content using a chat interface.

## Features
- Upload and process `.txt` documents.
- Add document content via a textarea.
- Query documents with natural language questions.

## Project Structure
Rag/ # Django app directory │ 
├── apiRag/ 
   ├── models.py # Database models │ 
   ├── utils.py # RAGPipeline for AI processing │ 
   ├── views.py # View logic │ 
   └── urls.py # App-specific URLs 
├── Rag/ # Django project directory │ 
   ├── settings.py # Project settings │ 
   ├── urls.py # Project URLs │ 
   └── wsgi.py # WSGI configuration 
├── faiss_index/ # FAISS vector index (ignored in Git) 
├── media/ # Uploaded files (ignored in Git) 
├── models/ # AI models (ignored in Git) 
├── static/ # Static files (ignored in Git) 
├── staticfiles/ # Collected static files (ignored in Git) 
├── templates/ # HTML templates │ 
    └── index.html # Main template 
├── manage.py # Django management script 
├── requirements.txt # Python dependencies 
├── Dockerfile # Docker image configuration 
├── docker-compose.yml # Docker Compose configuration 
├── .gitignore # Git ignore file 
├── .env # Environment variables (ignored in Git) 
    └── README.md # This file

## Setup Instructions
```bash
- 1. Clone the Repository
     git clone https://github.com/reskyagus21/ChatbotRAG.git
     cd your_repository

- 2. Prepare AI Model
     Automatic Download for bge-m3 & ms-marco-MiniLM-L6-v2:
     The docker-compose.yml will automatically download bge-m3 and ms-marco-MiniLM-L6-v2 from Hugging Face when you run the application.
     Download Manual Model mistral-7b-instruct-v0.2.Q4_K_M.gguf in https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/tree/main
     Choose the models and Place it in directory models/mistral-7b-instruct-v0.2.Q4_K_M.gguf

- 3. Create .env file
     DJANGO_SECRET_KEY=your-secret-key-here ( Replace your-secret-key-here with a secure Django secret key )
     DJANGO_SETTINGS_MODULE=Rag.settings
     DEBUG=False
     MYSQL_HOST=db
     MYSQL_DATABASE=rag
     MYSQL_USER=root
     MYSQL_PASSWORD=12345
     MYSQL_PORT=3306

- 4. Build and Run with Docker Compose
     docker-compose up --build

- 5.Access the Application
     Open http://localhost:8000 in your browser.
     Test the following features:
      - Upload a .txt document via the upload form.
      - Add text content via the textarea.
      - Ask questions through the chat interface.
```

## System Requirements
- **RAM**:
  - **Minimal**: 8GB (suitable for light testing, may be slow).
  - **Recommended**: 16GB or more for optimal performance.
  - **Optimal**: 32GB for heavy development or GPU-accelerated inference.



