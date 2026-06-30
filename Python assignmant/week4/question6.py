def convert_to_uppercase(words):
    return list(map(lambda w: w.upper(), words))


if __name__ == "__main__":
    print(convert_to_uppercase(["hello", "world"]))
