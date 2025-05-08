from django.db import models

class Document(models.Model):
    content = models.TextField()  # Isi dokumen asli
    vector_index_path = models.CharField(max_length=255)  # Path ke file indeks FAISS
    file = models.FileField(upload_to='documents/', null=True, blank=True)  # File dokumen
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Document {self.id}"
