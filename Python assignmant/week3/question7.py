def is_palindrome(word):
    normalized = word.lower()
    return normalized == normalized[::-1]


if __name__ == "__main__":
    print(is_palindrome("madam"))  # Output: True
    print(is_palindrome("hello"))  # Output: False
