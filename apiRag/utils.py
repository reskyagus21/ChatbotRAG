import os
import torch
import numpy as np
import logging
import re
import string
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import CrossEncoder
from django.conf import settings
from llama_cpp import Llama
from langchain.llms import LlamaCpp
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

# Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# Fungsi preprocessing teks
def remove_non_bmp_chars(text):
    return ''.join(c for c in text if ord(c) <= 0xFFFF)

def filter_documents(documents, query, min_length=50, max_length=1000):
    query_keywords = set(text(query).split())
    filtered = []
    for doc in documents:
        if min_length <= len(doc.page_content) <= max_length:
            doc_keywords = set(text(doc.page_content).split())
            if query_keywords.intersection(doc_keywords):
                filtered.append(doc)
    return filtered

def advanced_clean_output(text):
    text = re.sub(r'\b(\w+)( \1\b)+', r'\1', text)  # Hapus pengulangan kata
    sentences = text.split('. ')
    unique_sentences = []
    for sentence in sentences:
        if sentence and sentence not in unique_sentences:
            unique_sentences.append(sentence)
    return '. '.join(unique_sentences).strip()

class RAGPipeline:
    def __init__(self):
        try:
            self.embedding_model = HuggingFaceEmbeddings(
                model_name=os.path.join(settings.BASE_DIR, 'models/bge-m3'),
                model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
                encode_kwargs={"normalize_embeddings": True}
            )
            logger.info("Embedding model initialized.")
        except Exception as e:
            logger.error(f"Embedding model error: {e}")
            raise

        try:
            self.reranker = CrossEncoder(
                os.path.join(settings.BASE_DIR, 'models/ms-marco-MiniLM-L6-v2'),
            )
            logger.info("Reranker initialized.")
        except Exception as e:
            logger.error(f"Reranker error: {e}")
            raise

        try:
            # Konfigurasi LlamaCpp dengan Mistral-7B-Instruct-v0.2-GGUF
            model_path = os.path.join(settings.BASE_DIR, 'models/mistral-7b-instruct-v0.2.Q4_K_M.gguf')
            callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
            self.llm = LlamaCpp(
                model_path=model_path,
                n_ctx=2048,  # Panjang konteks maksimum
                max_tokens=200,  # Maksimum token keluaran
                temperature=0.7,  # Kontrol kreativitas
                top_p=0.9,  # Kontrol probabilitas kumulatif
                n_gpu_layers=40 if torch.cuda.is_available() else 0,  # Offload ke GPU jika tersedia
                f16_kv=True,  # Optimasi memori
                callback_manager=callback_manager,
                verbose=True
            )
            logger.info("Mistral-7B generator initialized.")
        except Exception as e:
            logger.error(f"LLM init error: {e}")
            raise

        self.vectorstore = None
        self.qa_chain = None
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            length_function=len
        )
        self.prompt_template = """
        [INST] Anda adalah asisten AI berbahasa Indonesia yang cerdas dan profesional. Jawablah pertanyaan berikut dengan teks yang di sediakan secara **ringkas**, **jelas**, dan **tanpa pengulangan**. Hindari jawaban yang tidak relevan, typo, atau bertele-tele. Jika tidak ada informasi yang cukup, jawab dengan: "Informasi tidak cukup untuk menjawab pertanyaan ini."

        Konteks: {context}

        Pertanyaan: {question}

        Jawaban yang diharapkan: Jawaban yang singkat, ada dalam teks dan langsung ke poin, tanpa pengulangan kata atau frasa.

        Jawaban: [/INST]
        """

    def create_index(self, texts, index_path):
        try:
            chunks = []
            for text in texts:
                cleaned_text = remove_non_bmp_chars(text)
                if cleaned_text:  # Pastikan teks tidak kosong setelah preprocessing
                    chunks.extend(self.text_splitter.split_text(cleaned_text))
            
            if not chunks:
                raise ValueError("Tidak ada teks valid untuk membuat indeks setelah preprocessing.")
            
            self.vectorstore = FAISS.from_texts(
                texts=chunks,
                embedding=self.embedding_model
            )
            os.makedirs(os.path.dirname(index_path), exist_ok=True)
            self.vectorstore.save_local(os.path.dirname(index_path), os.path.basename(index_path))
            return self.vectorstore
        except Exception as e:
            logger.error(f"Create index error: {e}")
            raise

    def load_index(self, index_path):
        try:
            self.vectorstore = FAISS.load_local(
                os.path.dirname(index_path),
                embeddings=self.embedding_model,
                index_name=os.path.basename(index_path),
                allow_dangerous_deserialization=True
            )
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=self.vectorstore.as_retriever(search_kwargs={"k": 5}),
                return_source_documents=True,
                chain_type_kwargs={"prompt": self._create_prompt()}
            )
        except Exception as e:
            logger.error(f"Load index error: {e}")
            raise

    def _create_prompt(self):
        return PromptTemplate(
            input_variables=["context", "question"],
            template=self.prompt_template
        )

    def rerank_documents(self, query, documents, threshold=0.5, max_docs=10):
        """Mengurutkan ulang dokumen berdasarkan relevansi dengan query."""
        try:
            filtered_docs = filter_documents(documents, query)[:max_docs]
            if not filtered_docs:
                logger.warning("Tidak ada dokumen yang lolos filter awal.")
                return documents[:3]
            
            pairs = [[query, doc.page_content] for doc in filtered_docs]
            scores = self.reranker.predict(pairs)
            
            valid_indices = [i for i, score in enumerate(scores) if score >= threshold]
            if not valid_indices:
                logger.warning("Tidak ada dokumen dengan skor di atas threshold.")
                return filtered_docs[:3]
            
            sorted_indices = sorted(valid_indices, key=lambda i: scores[i], reverse=True)
            reranked = [filtered_docs[i] for i in sorted_indices[:3]]
            return reranked
        except Exception as e:
            logger.error(f"Rerank error: {e}")
            return documents[:3]

    def query(self, question):
        try:
            if not self.qa_chain:
                raise ValueError("Index belum dimuat.")
            if not question or not isinstance(question, str):
                return "Pertanyaan tidak valid."

            cleaned_question = remove_non_bmp_chars(question)
            if not cleaned_question:
                return "Pertanyaan tidak valid setelah preprocessing."
            
            result = self.qa_chain.invoke({"query": cleaned_question})
            source_docs = result.get("source_documents", [])
            if not source_docs:
                return "Informasi tidak ditemukan."

            reranked_docs = self.rerank_documents(cleaned_question, source_docs, threshold=0.5)
            context = " ".join([doc.page_content for doc in reranked_docs])
            context = context[:1000] if len(context) > 1000 else context

            formatted_prompt = self.prompt_template.format(context=context, question=question)
            final_result = self.llm(formatted_prompt)

            return advanced_clean_output(final_result.strip()) if final_result else "Model tidak memberikan jawaban."
        except Exception as e:
            logger.error(f"Query error: {e}")
            return f"Terjadi kesalahan: {str(e)}"