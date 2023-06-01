from django.urls import path

from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('run/', views.run_task, name='runTask'),
    # path('call/', views.call_task, name='callTask')
]
