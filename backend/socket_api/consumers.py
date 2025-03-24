import json

from channels.generic.websocket import WebsocketConsumer
from channels.exceptions import StopConsumer
from utils.system_usage import get_system_usage

class ChatConsumer(WebsocketConsumer):
    """
    websocket类
    """
    def websocket_connect(self, message):
        """
        响应websocket连接
        Args:
            message:

        Returns:

        """
        self.accept()

    def websocket_receive(self, message):
        """
        接收websocket信息，并且根据信息内容发送响应报文
        Args:
            message:

        Returns:

        """
        if message['text'] == 'usage':
            usage = get_system_usage()
            response = json.dumps(usage)
            self.send(response)

    def websocket_disconnect(self, message):
        """
        websocket断开连接逻辑
        Args:
            message:

        Returns:

        """
        print('断开连接')
        raise StopConsumer