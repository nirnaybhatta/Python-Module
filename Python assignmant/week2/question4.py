def repeated_word_counts(words):
    frequency = {}
    duplicates = {}
    for word in words:
        frequency[word] = frequency.get(word, 0) + 1

    for word, count in frequency.items():
        if count > 1:
            duplicates[word] = count
    return duplicates


if __name__ == "__main__":
    words = input("Enter words separated by space: ").split()
    print(repeated_word_counts(words))
