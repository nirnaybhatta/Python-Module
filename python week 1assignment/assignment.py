# Question 1
num = float(input("Enter a decimal number: "))

num_int = int(num)
num_str = str(num)

print(f"Original float: {num}")
print(f"Converted to integer: {num_int}")
print(f'Converted to string: "{num_str}"')

# Question 2
full_name = input("Enter your full name: ")

names = full_name.split()

first_initial = names[0][0]
last_initial = names[-1][0]

print(f"Your initials are: {first_initial.upper()}.{last_initial.upper()}")

# Question 3
text = input("Enter a string: ")

reversed_text = text[::-1]

print(f"Reversed string: {reversed_text}")

# Question 4
word = input("Enter a word: ")
index = int(input("Enter starting index: "))

substring = word[index:]

print(f'Substring from index {index}: "{substring}"')


# Question 5
email = input("Enter an email address: ")

at_index = email.index('@')
domain = email[at_index + 1:]

print(f"Domain: {domain}")

# Question 6
word = input("Enter a word: ")

if len(word) > 1:
    new_word = word[-1] + word[1:-1] + word[0]
else:
    new_word = word

print(f"New word: {new_word}")


# Question 7
numbers = list(map(int, input("Enter numbers separated by space: ").split()))

for num in numbers:
    if num > 50:
        break
    if num % 5 == 0:
        continue
    print(num)


 # Question 8
password = input("Enter your password: ")

has_letter = any(c.isalpha() for c in password)
has_number = any(c.isdigit() for c in password)
has_special = any(c in "@#$%&" for c in password)

if len(password) < 6 or (has_letter and not has_number and not has_special):
    print("Weak Password")
elif len(password) >= 8 and has_letter and has_number and has_special:
    print("Strong Password")
elif len(password) >= 6 and has_letter and has_number:
    print("Moderate Password")
else:
    print("Weak Password")

 # Question 9
sentence = input("Enter a sentence: ")

words = sentence.split()

for i in range(len(words)):
    if i % 2 == 1:
        words[i] = words[i][::-1]

result = " ".join(words)

print(result)