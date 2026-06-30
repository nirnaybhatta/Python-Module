def write_to_file(file_path, content):
    with open(file_path, "w", encoding="utf-8") as file:
        file.write(content)


if __name__ == "__main__":
    write_to_file("output.txt", "This is a test")
    with open("output.txt", "r", encoding="utf-8") as file:
        print(file.read())
