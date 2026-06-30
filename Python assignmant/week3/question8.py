def char_count(s):
    counts = {}
    for char in s:
        counts[char] = counts.get(char, 0) + 1
    return counts


if __name__ == "__main__":
    print(char_count("hello"))  # Output: {'h': 1, 'e': 1, 'l': 2, 'o': 1}
