def filter_even_length_words(words):
    return list(filter(lambda w: len(w) % 2 == 0, words))


if __name__ == "__main__":
    print(filter_even_length_words(["hello", "world", "python", "is"]))
