import psutil
import pynvml


def get_system_usage():
    """
    获取系统信息对应的使用率，返回cpu使用率，内存使用率等
    Returns:

    """
    cpu_percent = psutil.cpu_percent(interval=0.5) / 100
    mem = psutil.virtual_memory()
    mem_percent = mem.percent / 100
    mem_used = mem.used / 1024 / 1024 / 1024
    mem_total = mem.total / 1024 / 1024 / 1024

    gpus = []
    try:
        import pynvml
        try:
            pynvml.nvmlInit()
            device_count = pynvml.nvmlDeviceGetCount()

            for i in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                gpu_info = {}

                try:
                    name = pynvml.nvmlDeviceGetName(handle)
                    gpu_info['name'] = name
                except pynvml.NVMLError:
                    gpu_info['name'] = f'GPU-{i}'

                # 获取gpu利用率
                try:
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    gpu_info['gpu_percent'] = util.gpu
                except pynvml.NVMLError:
                    gpu_info['gpu_percent'] = None

                # 获取显存使用情况
                try:
                    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    gpu_info['mem_used'] = mem_info.used / 1024 / 1024 / 1024
                    gpu_info['mem_total'] = mem_info.total / 1024 / 1024 / 1024
                except pynvml.NVMLError:
                    gpu_info['mem_used'] = None
                    gpu_info['mem_total'] = None

                gpus.append(gpu_info)
        except pynvml.NVMLError as e:
            pass
    except ImportError as e:
        pass


    return {
        'cpu_percent': cpu_percent,
        'mem_percent': mem_percent,
        'mem_used': mem_used,
        'mem_total': mem_total,
        'gpus': gpus
    }

if __name__ == '__main__':
    print(get_system_usage())