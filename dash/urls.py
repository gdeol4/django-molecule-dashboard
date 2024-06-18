# dash/urls.py
from django.urls import path
from . import views

app_name = 'dash'

urlpatterns = [
    path('', views.index, name='index'),
    path('ranking/', views.ranking_view, name='ranking'),
    path('rank/', views.rank, name='rank'),
    path('rank/<str:protein_target>/', views.protein_target_detail, name='protein_target_detail'),
    path('molecule/<str:inchikey>/', views.molecule_dashboard, name='molecule_dashboard'),
]