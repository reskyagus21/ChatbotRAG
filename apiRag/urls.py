from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('chat/', views.chat, name='chat'),
    path('upload/', views.upload_document, name='upload_document'),
    path('add_document_via_text/', views.add_document_via_text, name='add_document_via_text'),
]
