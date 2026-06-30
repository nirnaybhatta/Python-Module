def count_vowels(text):
    vowels = "aeiou"
    return sum(1 for char in text.lower() if char in vowels)


if __name__ == "__main__":
    print(count_vowels("Hello World"))  # Output: 3
