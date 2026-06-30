def count_word_frequency(words):
    frequency = {}
    for word in words:
        lowercase_word = word.lower()
        frequency[lowercase_word] = frequency.get(lowercase_word, 0) + 1
    return frequency


if __name__ == "__main__":
    words = input("Enter words separated by space: ").split()
    print(count_word_frequency(words))
