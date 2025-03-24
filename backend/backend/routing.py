from django.urls import path
from socket_api import consumers

websocket_urlpatterns = [
    path('ws/', consumers.ChatConsumer.as_asgi()),
]