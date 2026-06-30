class Student:
    def __init__(self, name, age, grades):
        self.name = name
        self.age = age
        self.grades = grades

    def average_grade(self):
        if not self.grades:
            return 0
        return sum(self.grades) / len(self.grades)


if __name__ == "__main__":
    students = [
        Student("Alice", 20, [75, 82, 91]),
        Student("Bob", 22, [65, 58, 72]),
        Student("Charlie", 19, [48, 52, 46])
    ]
    for student in students:
        print(f"{student.name}: {student.average_grade():.2f}")
