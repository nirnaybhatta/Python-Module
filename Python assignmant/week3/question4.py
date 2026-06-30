def largest_word(sentence):
    words = sentence.split()
    if not words:
        return ""
    longest = words[0]
    for word in words[1:]:
        if len(word) > len(longest):
            longest = word
    return longest


if __name__ == "__main__":
    print(largest_word("Python programming is awesome"))  # Output: programming
