kenalbulle_price = 35.00
discount = 0.60

num_buns = int(input("Enter the number of day-old cinnamon buns: "))

regular_price = num_buns * kenalbulle_price
discount_amount = regular_price * discount
total_price = regular_price - discount_amount


print(f"Regular Price:   {regular_price} SEK")
print(f"Discount:        {discount_amount} SEK")
print(f"Total Price:     {total_price} SEK")
