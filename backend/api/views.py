import os.path

from django.http import HttpResponse, JsonResponse
from django.shortcuts import render
from django.views.decorators.http import require_http_methods
from utils import *
import json



# Create your views here.
@require_http_methods(["GET"])
def view_is_train_running(request):
    """
    是否正在训练
    Args:
        request:

    Returns:

    """
    response = {}
    train_file_name = 'cdfss.py'
    try:
        ret = is_program_running(train_file_name)
        response['err_code'] = 0
        response['msg'] = str(ret)
    except Exception as e:
        response['err_code'] = 1
        response['msg'] = str(e)
    return JsonResponse(response)

@require_http_methods(["GET"])
def view_get_dataset_list(request):
    """
    获取数据集列表
    Args:
        request:

    Returns:

    """
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
    """
    获取权重列表
    Args:
        request:

    Returns:

    """
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
    """
    配置文件上传之后的逻辑
    Args:
        request:

    Returns:

    """
    config_save_path = "../config/temp.json"
    response = {}

    try:
        config_content = json.loads(request.body)
        config_content['dataset']["path"] = os.path.join('./data/dataset/',config_content['dataset']["path"])
        config_content['train']['pretrained']['weight_path'] = (
            os.path.join('./data/weights/',config_content['train']['pretrained']['weight_path']))

        with open(config_save_path, "w", encoding='utf-8') as f:
            json.dump(config_content, f, ensure_ascii=False )

        pid = launch_detached_script()
        response["err_code"] = 0
        response["msg"] = f"{str(pid)}"
    except Exception as e:
        response['err_code'] = 1
        response['msg'] = str(e)

    return JsonResponse(response)

@require_http_methods(["GET"])
def view_get_logs_list(request):
    """
    获取日志列表
    Args:
        request:

    Returns:

    """
    response = {}
    try:
        logs_list = get_logs_list()
        response['err_code'] = 0
        response['msg'] = logs_list
    except Exception as e:
        response['err_code'] = 1
        response['msg'] = str(e)

    return JsonResponse(response)

@require_http_methods(["GET"])
def view_is_tensorboard_running(request):
    """
    tensorboard是否正在运行
    Args:
        request:

    Returns:

    """
    response = {}
    try:
        pid = target_port_used(6006)
        if pid == -1:
            response['err_code'] = 0
            response['msg'] = "False"
        else:
            response['err_code'] = 0
            response['msg'] = "True"
    except Exception as e:
        response['err_code'] = 1
        response['msg'] = str(e)

    return JsonResponse(response)

@require_http_methods(["GET"])
def view_start_tensorboard(request):
    """
    启动tensorboard
    Args:
        request:

    Returns:

    """
    response = {}
    try:
        res = start_tensorboard()
        if res:
            response['err_code'] = 0
            response['msg'] = "True"
        else:
            response['err_code'] = 0
            response['msg'] = "False"
    except Exception as e:
        response['err_code'] = 1
        response['msg'] = str(e)

    return JsonResponse(response)

@require_http_methods(["GET"])
def view_end_tensorboard(request):
    """
    结束tensorboard
    Args:
        request:

    Returns:

    """
    response = {}
    try:
        res = kill_process_on_port(6006)
        if res:
            response['err_code'] = 0
            response['msg'] = "True"
        else:
            response['err_code'] = 0
            response['msg'] = "False"
    except Exception as e:
        response['err_code'] = 1
        response['msg'] = str(e)

    return JsonResponse(response)

@require_http_methods(["GET"])
def view_end_training(request):
    """
    结束训练
    Args:
        request:

    Returns:

    """
    response = {}
    try:
        res = end_training()
        if res:
            response['err_code'] = 0
            response['msg'] = "True"
        else:
            response['err_code'] = 0
            response['msg'] = "False"
    except Exception as e:
        response['err_code'] = 1
        response['msg'] = str(e)

    return JsonResponse(response)