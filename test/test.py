# import psutil
#
# def is_program_running(target_name):
#     for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
#         try:
#             # 检查是否是Python进程，且命令行包含目标名称
#             if 'python' in proc.info['name'].lower():
#                 cmdline = proc.info['cmdline']
#                 if len(cmdline) >= 1 and target_name in cmdline:
#                     # 排除当前进程自身
#                     print(cmdline)
#                     if proc.pid != psutil.Process().pid:
#                         return True
#         except (psutil.NoSuchProcess, psutil.AccessDenied):
#             continue
#     return False
#
#
#
# if __name__ == '__main__':
#     if is_program_running('hello.py'):
#         print("程序正在运行")
#     else:
#         print("程序未运行")
