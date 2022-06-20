from django.urls import path
from classificationmodel import views
from django.conf.urls.static import static
from django.conf import settings

urlpatterns = [
    path('', views.index, name='index'),
    path('predictclass', views.predictclass, name='predictclass'),
]

urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)