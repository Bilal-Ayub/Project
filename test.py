class Person:
    def __init__(self, name: str, age: int, gender: str) -> None:
        self.name = name
        self.age = age
        self.gender = gender

    def print_info(self):
        print("Name:", self.name)
        print("Age: ", self.age, "baras")
        # print("Gender:", self.gender)


class Teacher(Person):
    def __init__(self, name: str, age: int, gender: str, designation: str) -> None:
        self.designation = designation
        super().__init__(name, age, gender)

    def print_info(self):
        super().print_info()
        print("Designation: ", self.designation)


class Student(Person):
    def __init__(self, name: str, age: int, gender: str, major: str) -> None:
        self.major = major
        super().__init__(name, age, gender)

    def print_info(self):
        super().print_info()
        print("Major: ", self.major)


areeba = Student(name="Areeba", age=19, gender="Attack Helicopter", major="CS")
bilal = Student(name="Bilal", age=19, gender="Short Term Memory Loss", major="CS")
meesum = Teacher(
    name="Mirza Muhammad Meesum Ali Qazalbash", age=22, gender="Karen", designation="RA"
)

# print(areeba.name)
# print(bilal.name)
# print(meesum.age)

areeba.print_info()
print()
bilal.print_info()
print()
meesum.print_info()
