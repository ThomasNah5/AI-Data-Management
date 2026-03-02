import matplotlib.pyplot as plt
import numpy as np

# # Initialize data
# data = {"apple": 10, "orange": 15, "lemon": 5, "lime": 20}

# # Extract keys from the data
# names = list(data.keys())

# # Extract values from the data
# values = list(data.values())

# # Create a figure and subplots, with a grid with a row and column and sets the size of the figure
# fig, axs = plt.subplots(1, 3, figsize=(9, 3), sharey=True)

# # create a bar chart, scatter plot and line plot
# axs[0].bar(names, values)
# axs[1].scatter(names, values)
# axs[2].fill_between(names, values, alpha=0.7)

# for ax in axs[0], axs[1], axs[2]:
#     ax.grid(True)
# fig.suptitle("Categorical Plotting")

# plt.show()


# fig, ax = plt.subplots()

# fruits = ["apple", "blueberry", "cherry", "orange"]
# counts = [40, 100, 30, 55]
# bar_labels = ["red", "blue", "_red", "orange"]
# bar_colors = ["tab:red", "tab:blue", "tab:red", "tab:orange"]

# ax.bar(fruits, counts, label=bar_labels, color=bar_colors)
# ax.set_ylabel("fruit supply")
# ax.set_title("Fruit supply by kind and color")
# ax.legend(title="Fruit color")

# plt.show()


# data from United Nations World Population Prospects (Revision 2019)
# https://population.un.org/wpp/, license: CC BY 3.0 IGO
year = [1950, 1960, 1970, 1980, 1990, 2000, 2010, 2018]
population_by_continent = {
    "africa": [228, 284, 365, 477, 631, 814, 1044, 1275],
    "americas": [340, 425, 519, 619, 727, 840, 943, 1006],
    "asia": [1394, 1686, 2120, 2625, 3202, 3714, 4169, 4560],
    "europe": [220, 253, 276, 295, 310, 303, 294, 293],
    "oceania": [12, 15, 19, 22, 26, 31, 36, 39],
}

fig, ax = plt.subplots()
ax.stackplot(
    year,
    list(population_by_continent.values()),
    labels=population_by_continent.keys(),
    alpha=0.8,
)
ax.legend(loc="upper left", reverse=True)
ax.set_title("World population")
ax.set_xlabel("Year")
ax.set_ylabel("Number of people (millions)")

plt.show()
