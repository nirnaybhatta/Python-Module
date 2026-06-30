class StudentResult:
    def __init__(self, name, age, average_grade):
        self.name = name
        self.age = age
        self.average_grade = average_grade

    def has_passed(self):
        return self.average_grade >= 50


if __name__ == "__main__":
    results = [
        StudentResult("Alice", 20, 82.67),
        StudentResult("Bob", 22, 65.0),
        StudentResult("Charlie", 19, 48.67)
    ]
    for result in results:
        if result.has_passed():
            print(result.name)
