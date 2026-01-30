kenalbulle_price = 35.00
discount_rate = 0.60

# Read input from the user
try:
    num_buns = int(input("Enter the number of day-old cinnamon buns: "))
except ValueError:
    print("Invalid input. Please enter a whole number.")
    exit()

regular_price = num_buns * kenalbulle_price
discount_amount = regular_price * discount_rate
total_price = regular_price - discount_amount

# Display the results with aligned decimal points
print("\n--- Receipt ---")
print(f"Regular Price:   {regular_price:10.2f} SEK")
print(f"Discount:        {discount_amount:10.2f} SEK")
print(f"Total Price:     {total_price:10.2f} SEK")
