import os
import io
import time
from django.conf import settings
from django.core.files.storage import FileSystemStorage
from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from .models import Document
from .utils import RAGPipeline, remove_non_bmp_chars

# Inisialisasi pipeline RAG hanya sekali untuk efisiensi
rag_pipeline = RAGPipeline()


def index(request):
    """Render halaman utama aplikasi."""
    return render(request, 'index.html')

@csrf_exempt
def chat(request):
    """
    Menangani permintaan POST untuk memproses pertanyaan pengguna menggunakan RAGPipeline.
    Mengembalikan jawaban atau pesan error dalam format JSON.
    """
    if request.method != "POST":
        return JsonResponse({"error": "Metode tidak diizinkan."}, status=405)

    question = request.POST.get("question")
    
    # Validasi pertanyaan
    if not question or not isinstance(question, str):
        return JsonResponse({"error": "Pertanyaan tidak valid."}, status=400)

    try:
        # Ambil semua dokumen dari database
        documents = Document.objects.all()
        if not documents:
            return JsonResponse({"response": "Tidak ada dokumen yang tersedia."}, status=200)

        # Gabungkan teks dari semua dokumen
        texts = [doc.content for doc in documents]
        first_index_path = os.path.join(settings.BASE_DIR, documents[0].vector_index_path)

        # Buat indeks jika belum ada
        if not os.path.exists(first_index_path):
            rag_pipeline.create_index(texts, first_index_path)

        # Muat indeks
        rag_pipeline.load_index(first_index_path)

        # Proses pertanyaan dan dapatkan jawaban
        response = rag_pipeline.query(question)
        return JsonResponse({"response": response}, status=200)

    except ValueError as ve:
        return JsonResponse({"error": str(ve)}, status=400)
    except Exception as e:
        return JsonResponse({"error": f"Terjadi kesalahan: {str(e)}"}, status=500)


@csrf_exempt
def upload_document(request):
    """
    Menangani permintaan POST untuk mengunggah dokumen teks (.txt).
    Memproses dokumen, membuat indeks FAISS, dan menyimpan metadata ke database.
    """
    if request.method != "POST" or not request.FILES.get("document"):
        return JsonResponse({"error": "Metode tidak diizinkan atau file tidak ditemukan."}, status=405)

    uploaded_file = request.FILES["document"]

    # Validasi ekstensi file
    if not uploaded_file.name.endswith('.txt'):
        return JsonResponse({"error": "Hanya file .txt yang diizinkan."}, status=400)

    try:
        # Baca dan bersihkan teks dari file
        text = uploaded_file.read().decode('utf-8', errors='ignore')
        cleaned_text = remove_non_bmp_chars(text)

        # Validasi teks tidak kosong setelah preprocessing
        if not cleaned_text:
            return JsonResponse({"error": "Dokumen tidak berisi teks valid setelah preprocessing."}, status=400)

        # Simpan file ke direktori media
        fs = FileSystemStorage(location=settings.MEDIA_ROOT)
        cleaned_file_io = io.BytesIO(cleaned_text.encode('utf-8'))
        filename = fs.save(uploaded_file.name, cleaned_file_io)

        # Buat indeks FAISS untuk dokumen
        index_dir = os.path.join(settings.BASE_DIR, 'faiss_index')
        os.makedirs(index_dir, exist_ok=True)
        index_path = os.path.join(index_dir, f'doc_{filename}.faiss')

        rag_pipeline.create_index([cleaned_text], index_path)

        # Simpan metadata dokumen ke database
        Document.objects.create(
            content=cleaned_text,
            vector_index_path=f'faiss_index/doc_{filename}.faiss',
            file=filename
        )

        return JsonResponse({"message": "Dokumen berhasil diunggah dan diindeks."}, status=200)

    except ValueError as e:
        return JsonResponse({"error": str(e)}, status=400)
    except Exception as e:
        return JsonResponse({"error": f"Terjadi kesalahan: {str(e)}"}, status=500)
    

@csrf_exempt
def add_document_via_text(request):
    if request.method != "POST":
        return JsonResponse({"error": "Metode tidak diizinkan."}, status=405)

    text_content = request.POST.get("text_content")
    if not text_content or not isinstance(text_content, str):
        return JsonResponse({"error": "Teks dokumen tidak valid."}, status=400)

    try:
        cleaned_text = remove_non_bmp_chars(text_content)
        if not cleaned_text:
            return JsonResponse({"error": "Teks dokumen tidak valid setelah preprocessing."}, status=400)

        fs = FileSystemStorage(location=settings.MEDIA_ROOT)
        filename = f"document_{int(time.time())}.txt"
        with fs.open(filename, 'w') as file:
            file.write(cleaned_text)

        index_dir = os.path.join(settings.BASE_DIR, 'faiss_index')
        os.makedirs(index_dir, exist_ok=True)
        index_path = os.path.join(index_dir, f'doc_{filename}.faiss')

        rag_pipeline.create_index([cleaned_text], index_path)

        Document.objects.create(
            content=cleaned_text,
            vector_index_path=f'faiss_index/doc_{filename}.faiss',
            file=filename
        )

        return JsonResponse({"message": "Dokumen berhasil ditambahkan dan diindeks."}, status=200)
    except ValueError as e:
        return JsonResponse({"error": str(e)}, status=400)
    except Exception as e:
        return JsonResponse({"error": f"Terjadi kesalahan: {str(e)}"}, status=500)