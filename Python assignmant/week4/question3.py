from question1 import read_file_content


def find_longest_word(file_path):
    content = read_file_content(file_path)
    words = content.split()
    if not words:
        return ""
    longest = words[0]
    for word in words[1:]:
        if len(word) > len(longest):
            longest = word
    return longest


if __name__ == "__main__":
    sample_path = "sample.txt"
    with open(sample_path, "w", encoding="utf-8") as file:
        file.write("Hello amazing world")
    print(find_longest_word(sample_path))
