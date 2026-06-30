from question1 import read_file_content
from question2 import write_to_file


def reverse_file_content(file_path):
    content = read_file_content(file_path)
    reversed_content = content[::-1]
    write_to_file("reversed.txt", reversed_content)
    return "reversed.txt"


if __name__ == "__main__":
    sample_path = "sample.txt"
    write_to_file(sample_path, "Hello world")
    print(reverse_file_content(sample_path))
