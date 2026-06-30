def alternate_word_reverse(text):
    words = text.split()
    for index in range(1, len(words), 2):
        words[index] = words[index][::-1]
    return " ".join(words)


if __name__ == "__main__":
    text = input("Enter a sentence: ")
    print(alternate_word_reverse(text))
