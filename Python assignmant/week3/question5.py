def sum_of_list(numbers):
    total = 0
    for num in numbers:
        total += num
    return total


if __name__ == "__main__":
    print(sum_of_list([1, 2, 3, 4, 5]))  # Output: 15
