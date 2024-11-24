import os


class DataAnnotation:
    def __init__(self, data_path: str):
        self.data_path = data_path

        if not os.path.exists(self.data_path):
            print(f"'{data_path}' is not exists.")
            exit(1)
        if not os.path.isdir(self.data_path):
            print(f"'{data_path}' is not a directory.")
            exit(1)


    def do_annotation(self):
        pass