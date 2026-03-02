# Pandas

import pandas as pd

# Creating a DataFrame
actors_df = pd.read_csv("actors.csv", header=None)
# print(actors_df)

# Save the DataFrame to a CSV file
actors_df.to_csv("actors.csv", index=False)


# Reload the DataFrame from the CSV file
reloaded_df = pd.read_csv("actors.csv")

# Display the reloaded DataFrame to verify it matches the original
# print(reloaded_df)

print(reloaded_df.iloc[7:8, :])
# X_l = reloaded_df.iloc[:, 1:-1].values # features set
# y_p = reloaded_df.iloc[:, -1].values # set of study variable

# for selecting every 5 row:rows devidable on 5
print(reloaded_df.iloc[lambda x: x.index % 5 == 0])

# or use Slice [start:stop:step], starting from index 5 showing every 6th row in the dataset -> which will be (5,11,17,23,....)
# print(reloaded_df= reloaded_df[5::6])
