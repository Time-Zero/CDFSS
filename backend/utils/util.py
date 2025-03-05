import psutil
import os
import subprocess
import platform
import sys

def is_program_running(target_name):
    """
    判断是否有对应文件名称的python程序正在运行
    Args:
        target_name: 要检查的python文件名称

    Returns:

    """
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            # 检查是否是Python进程，且命令行包含目标名称
            if 'python' in proc.info['name'].lower():
                cmdline = proc.info['cmdline']
                if len(cmdline) >= 1 and target_name in cmdline:
                    # 排除当前进程自身
                    print(cmdline)
                    if proc.pid != psutil.Process().pid:
                        return True
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return False

def get_dataset_list():
    """
    获取数据集列表
    Returns:

    """
    dataset_dir = "../data/dataset"
    try:
        res = []
        datasets_list = os.listdir(dataset_dir)
        for dataset_name in datasets_list:
            if os.path.isdir(os.path.join(dataset_dir, dataset_name)):
                res.append(dataset_name)

        return res;
    except Exception as e:
        return {'err': str(e)};

def get_weight_list():
    weights_dir = "../data/weights"
    try:
        res = []
        weights_list = sorted(os.listdir(weights_dir))
        for weight_name in weights_list:
            if (os.path.isfile(os.path.join(weights_dir, weight_name))
                    and weight_name.lower().endswith(".pth")):
                res.append(weight_name)

        return res
    except Exception as e:
        return {'err': str(e)};

def start_train():
    import subprocess
    import platform

    script_path = "../../cdfss.py"
    script_args = "-c ../../temp.json"
    target_dir = "../../"

    if platform.system() in ['Linux', 'Darwin']:  # Linux/Mac
        cmd = f"nohup python {script_path} {script_args} > /dev/null 2>&1 &"
        process = subprocess.Popen(cmd, shell=True,cwd=target_dir)
    else:  # Windows
        cmd = f'start /B python {script_path} {script_args}'
        process = subprocess.Popen(cmd, shell=True,cwd=target_dir)

    print(f"pid: {process.pid}")

if __name__ == '__main__':
    start_train()