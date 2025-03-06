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
                if cmdline is not None and len(cmdline) >= 1 and target_name in cmdline:
                    # 排除当前进程自身
                    if proc.pid != psutil.Process().pid:
                        return proc.pid
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return 0


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


def launch_detached_script():
    """
        以独立终端的方式启动训练脚本
    Returns:
        pid(int): 返回训练脚本的pid
    """
    # 切换到目标目录
    cur_dir = os.getcwd()
    target_dir = "../"
    os.chdir(target_dir)

    # 脚本名称及参数
    script_name = "cdfss.py"
    script_args = ["-c", "config/temp.json"]
    cmd = [sys.executable, script_name] + script_args

    # 日志文件配置
    log_file = "output.log"  # 输出文件名
    # 以追加模式打开文件，并启用行缓冲
    with open(log_file, "a", buffering=1) as f:  # buffering=1 表示行缓冲
        if sys.platform.startswith('win'):
            # Windows: 创建新进程组并重定向输出
            proc = subprocess.Popen(
                cmd,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
                stdout=f,
                stderr=subprocess.STDOUT,  # 合并标准错误到输出
                stdin=subprocess.DEVNULL  # 禁止标准输入
            )
        else:
            # Unix: 创建新会话并重定向输出
            proc = subprocess.Popen(
                cmd,
                start_new_session=True,
                stdout=f,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL
            )

    os.chdir(cur_dir)
    return proc.pid


if __name__ == '__main__':
    # start_train()
    pass
