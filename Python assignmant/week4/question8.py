def process_file_with_lambda(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        lines = file.readlines()

    processed_lines = []
    for line in lines:
        processed_words = list(map(lambda word: word.upper(), line.split()))
        processed_lines.append(" ".join(processed_words))

    with open(file_path, "w", encoding="utf-8") as file:
        file.write("\n".join(processed_lines))


if __name__ == "__main__":
    sample_path = "sample.txt"
    with open(sample_path, "w", encoding="utf-8") as file:
        file.write("Hello world\nThis is python")
    process_file_with_lambda(sample_path)
    with open(sample_path, "r", encoding="utf-8") as file:
        print(file.read())
