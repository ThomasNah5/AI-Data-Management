def is_prime_number(number):
    if number < 2:
        return False
    for i in range(2, number):
        if number % i == 0:
            return False

    return True


def main():
    num = int(input("Enter number: "))
    if is_prime_number(num):
        print(f"{num} is a prime number")
    else:
        print(f"{num} is not a prime number")


main()
