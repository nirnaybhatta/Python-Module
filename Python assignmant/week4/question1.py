import os


def read_file_content(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()


if __name__ == "__main__":
    sample_path = "sample.txt"
    with open(sample_path, "w", encoding="utf-8") as file:
        file.write("Hello world")
    print(read_file_content(sample_path))
