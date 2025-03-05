import os.path

from django.http import HttpResponse, JsonResponse
from django.shortcuts import render
from django.views.decorators.http import require_http_methods
from utils.util import *
import json



# Create your views here.
@require_http_methods(["GET"])
def view_is_train_running(request):
    response = {}
    train_file_name = 'cdfss.py'
    try:
        ret = is_program_running(train_file_name)
        if ret:
            response['err_code'] = 0
            response['msg'] = 'True'
        else:
            response['err_code'] = 0
            response['msg'] = 'False'
    except Exception as e:
        response['err_code'] = 1
        response['msg'] = str(e)
    return JsonResponse(response)

@require_http_methods(["GET"])
def view_get_dataset_list(request):
    response = {}
    try:
        ret = get_dataset_list()
        if isinstance(ret, list):
            response['err_code'] = 0
            response['msg'] = ret;
        else:
            response['err_code'] = 1
            response['msg'] = ret['err']
    except Exception as e:
        response['err_code'] = 1
        response['msg'] = str(e)

    return JsonResponse(response)

@require_http_methods(["GET"])
def view_get_weight_list(request):
    response = {}
    try:
        ret = get_weight_list()
        if isinstance(ret, list):
            response['err_code'] = 0
            response['msg'] = ret;
        else:
            response['err_code'] = 1
            response['msg'] = ret['err']
    except Exception as e:
        response['err_code'] = 1
        response['msg'] = str(e)

    return JsonResponse(response)

@require_http_methods(["POST"])
def view_upload_form(request):
    config_save_path = "../config/temp.json"
    response = {}

    try:
        config_content = json.loads(request.body)
        config_content['dataset']["path"] = os.path.join('./data/dataset/',config_content['dataset']["path"])
        config_content['train']['pretrained']['weight_path'] = (
            os.path.join('./data/weights/',config_content['train']['pretrained']['weight_path']))

        with open(config_save_path, "w", encoding='utf-8') as f:
            json.dump(config_content, f, ensure_ascii=False )

        response["err_code"] = 0
        response["msg"] = ""
    except Exception as e:
        response['err_code'] = 1
        response['msg'] = str(e)

    return JsonResponse(response)
