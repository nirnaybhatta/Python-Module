import os


def check_file_empty(file_path):
    return os.path.exists(file_path) and os.path.getsize(file_path) == 0


if __name__ == "__main__":
    empty_path = "empty.txt"
    open(empty_path, "w", encoding="utf-8").close()
    print(check_file_empty(empty_path))
