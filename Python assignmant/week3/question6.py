def to_title_case(sentence):
    return " ".join(word.capitalize() for word in sentence.split())


if __name__ == "__main__":
    print(to_title_case("hello world from python"))  # Output: Hello World From Python
