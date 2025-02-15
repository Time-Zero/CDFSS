from colorama import Fore, Style


def func_print(color: str, star_num: int, content: str):
    if color == 'red':
        str_color = Fore.RED
    elif color == 'green':
        str_color = Fore.GREEN
    elif color == 'yellow':
        str_color = Fore.YELLOW
    elif color == 'blue':
        str_color = Fore.BLUE
    elif color == 'magenta':
        str_color= Fore.MAGENTA
    else :
        str_color = Style.BRIGHT

    print(str_color + star_num * '*' + content + star_num * '*' + Style.RESET_ALL)