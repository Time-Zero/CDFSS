from django.http import HttpResponse, JsonResponse
from django.shortcuts import render
from django.views.decorators.http import require_http_methods


# Create your views here.
@require_http_methods(["GET"])
def test(request):
    response = {"message": "test"}
    return JsonResponse(response)