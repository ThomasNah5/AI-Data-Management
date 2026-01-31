import random

flips = ""
consecutive = 0
last_flip = ""

while consecutive < 3:
    result = "H" if random.randint(0, 1) == 0 else "T"
    flips += result

    if result == last_flip:
        consecutive += 1
    else:
        consecutive = 1
        last_flip = result

print(flips)
print(f"Number of flips: {len(flips)}")
