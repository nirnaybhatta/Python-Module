class TextAnalyzer:
    def __init__(self, text):
        self.text = text

    def word_count(self):
        return len(self.text.split())


if __name__ == "__main__":
    analyzer = TextAnalyzer("Python programming is fun and educational")
    print(analyzer.word_count())
