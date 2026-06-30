def check_password_strength(password):
    special_chars = "@#$%&"
    has_letter = any(c.isalpha() for c in password)
    has_digit = any(c.isdigit() for c in password)
    has_special = any(c in special_chars for c in password)

    if len(password) < 6 or (has_letter and not has_digit and not has_special):
        return "Weak"
    if len(password) >= 8 and has_letter and has_digit and has_special:
        return "Strong"
    if len(password) >= 6 and has_letter and has_digit:
        return "Moderate"
    return "Weak"


if __name__ == "__main__":
    password = input("Enter a password: ")
    print(check_password_strength(password))
