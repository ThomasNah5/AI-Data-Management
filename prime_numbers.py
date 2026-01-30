def prime(n):
    if n < 2:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True


def main():
    numbers_to_check = [
        83,
        89,
        97,
    ]
    for num in numbers_to_check:
        if prime(num):
            print(f"{num} is a prime number")
        else:
            print(f"{num} is not a prime number")


main()
