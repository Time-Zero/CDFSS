from .views import *
from django.urls import path

urlpatterns = [
    path('running', view_is_train_running),
    path('ds_list', view_get_dataset_list),
    path('wt_list', view_get_weight_list),
    path('ul_form', view_upload_form)
]