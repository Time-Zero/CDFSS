def singleton(cls):
    """
    实现单例模式的装饰器
    :param cls: 传入的类
    :return: 类的唯一对象
    """
    instance = {}
    def get_instance(*args, **kwargs):
        if cls not in instance:
            instance[cls] = cls(*args, **kwargs)
        return instance[cls]

    return get_instance