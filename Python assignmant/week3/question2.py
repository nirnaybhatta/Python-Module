def find_maximum(list1):
    if not list1:
        raise ValueError("The list must not be empty")
    maximum = list1[0]
    for num in list1[1:]:
        if num > maximum:
            maximum = num
    return maximum


if __name__ == "__main__":
    print(find_maximum([10, 25, 5, 80, 30]))  # Output: 80
