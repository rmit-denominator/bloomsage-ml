import os


def clean_dir(root_dir: str = None) -> None:
    if root_dir is None:
        raise ValueError('root_dir must be provided')

    for root, dirs, files in os.walk(root_dir, topdown=False):
        for file in files:
            file_path = os.path.join(root, file)
            os.remove(file_path)
        for dir_name in dirs:
            dir_path = os.path.join(root, dir_name)
            for item in os.listdir(dir_path):
                item_path = os.path.join(dir_path, item)
                if os.path.isfile(item_path):
                    os.remove(item_path)
                else:
                    clean_dir(item_path)
            os.rmdir(dir_path)
