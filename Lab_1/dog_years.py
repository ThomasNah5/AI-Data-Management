human_years = float(input("Enter the dog's age in human years: "))

if human_years < 0:
    print("Error: Age cannot be negative.")
elif human_years <= 2:
    dog_years = human_years * 10.5
    print(f"The dog's age in dog years is {dog_years:.1f}")
else:
    dog_years = 2 * 10.5 + (human_years - 2) * 4
    print(f"The dog's age in dog years is {dog_years:.1f}")
