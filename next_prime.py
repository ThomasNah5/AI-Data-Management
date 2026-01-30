def is_prime_number(number):
    if number < 2:
        return False
    for i in range(2, number):
        if number % i == 0:
            return False
    return True


def nextPrime(n):
    number = n + 1

    while not is_prime_number(number):
        number += 1

    return number


def main():
    num = int(input("Enter an integer: "))

    if is_prime_number(num):
        print(f"{num} is a prime number.")
    else:
        print(f"{num} is not a prime number.")

    next_prime = nextPrime(num)
    print(f"The first prime number larger than {num} is {next_prime}.")


main()
