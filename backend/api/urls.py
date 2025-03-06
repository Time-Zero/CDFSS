from .views import *
from django.urls import path

urlpatterns = [
    path('running', view_is_train_running),
    path('ds_list', view_get_dataset_list),
    path('wt_list', view_get_weight_list),
    path('ul_form', view_upload_form),
    path('lg_list', view_get_logs_list),
    path('tb_running', view_is_tensorboard_running),
    path('start_tensorboard', view_start_tensorboard),
    path('end_tensorboard', view_end_tensorboard),
    path('end_training', view_end_training),
]