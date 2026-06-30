books = {
    "Book1": 5,
    "Book2": 6,
    "Book3": 10
}


def get_valid_copies():
    while True:
        user_input = input("Enter number of copies: ").strip()
        if user_input == "":
            print("Invalid input. Please enter a valid integer.")
            continue
        try:
            return int(user_input)
        except ValueError:
            print("Invalid input. Please enter a valid integer.")


if __name__ == "__main__":
    book_name = input("Enter the name of the book: ")
    copies = get_valid_copies()

    if book_name in books:
        if books[book_name] >= copies:
            print("Available")
        else:
            print("Partially Available")
    else:
        print("Unavailable")
