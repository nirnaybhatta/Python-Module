class Book:
    def __init__(self, title, author):
        self.title = title
        self.author = author

    def short_title(self):
        return self.title[:10]


if __name__ == "__main__":
    books = [
        Book("The Python Journey", "Author A"),
        Book("Learn Programming", "Author B"),
        Book("Code in Practice", "Author C")
    ]
    for book in books:
        print(book.short_title())
