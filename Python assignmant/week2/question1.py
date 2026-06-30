def skip_divisible_by_5(numbers):
    for num in numbers:
        if num > 50:
            break
        if num % 5 == 0:
            continue
        print(num)


if __name__ == "__main__":
    numbers = list(map(int, input("Enter numbers separated by space: ").split()))
    skip_divisible_by_5(numbers)
