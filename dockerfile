# Gunakan image Python resmi sebagai base image
FROM python:3.10-slim

# Set environment variables untuk optimasi Python
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Instal dependensi sistem untuk MySQL, FAISS, dan model AI
RUN apt-get update && apt-get install -y \
    build-essential \
    libatlas-base-dev \
    gfortran \
    default-libmysqlclient-dev \
    pkg-config \
    git \
    cmake \
    libopenblas-dev \
    libomp-dev \
    netcat-openbsd \
    && rm -rf /var/lib/apt/lists/*


RUN useradd -m -r appuser

# Buat direktori kerja
WORKDIR /app

# Upgrade pip dan instal dependensi Python
RUN pip install --upgrade pip

# Salin requirements.txt dan instal dependensi
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Salin script wait-for-db.sh dan beri izin eksekusi
COPY scripts/wait-for-db.sh /usr/local/bin/wait-for-db.sh

# Beri izin eksekusi untuk wait-for-db.sh
RUN chmod +x /usr/local/bin/wait-for-db.sh

COPY . .
# Buat direktori dan beri izin untuk media, FAISS, static
RUN mkdir -p /app/media /app/faiss_index /app/static /app/staticfiles && \
    chown -R appuser:appuser /app/media /app/faiss_index /app/static /app/staticfiles

# Tambahkan user non-root untuk keamanan
USER appuser

# Expose port untuk aplikasi
EXPOSE 8000

# Jalankan migrasi, kumpulkan file statis, dan jalankan Gunicorn
CMD ["sh", "-c", "python manage.py migrate && python manage.py collectstatic --noinput && gunicorn --bind 0.0.0.0:8000 --workers 3 Rag.wsgi:application"]
