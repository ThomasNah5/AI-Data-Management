# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd

# df = pd.read_csv("Data.csv")
# # # print(df.head())
# # print(df.tail())

# # # # df.info()
# # # # pd.isna(df)

# # # df.describe().transpose()
# # df.info()
# # pd.isna(df)
# # df.isna().sum()

# df_1 = df.dropna()
# # print(df_1)

# # Sort by country name
# # sorted = df.sort_values(by=["Country"])
# # print(sorted)

# # mean/median imputation - use this technique

# # df["Age"] = df["Age"].fillna(df["Age"].mean())
# # df["Salary"] = df["Salary"].fillna(df["Salary"].median())
# # # Viewing the dataframe
# # print(df)

# # Backward/Forward Fill

# df["Age"] = df["Age"].fillna(method="bfill")
# df["Salary"] = df["Salary"].fillna(method="ffill")
# # Viewing the dataframe
# print(df)

# cols = ["Age", "Salary"]  # selecting and saving the columns that need to be removed
# df_2 = df.drop(
#     cols, axis=1
# )  # removing the selected columns. axis 1 refers to columns and axis 0 refers to rows
# print(df_2)

# Noise Filtering(Supplementary)

# Replace/Add/Split Values


# Noise reduction - The purpose of noise reduction is to use signal processing methods to alter the data slightly and remove the noises from them. Some common methods are the Infinite Impulse Response (IIR) filter, Kalman Filter (KF), Fourier Transform, etc.
# Replace/Add/Split Values
from datetime import datetime

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

df = pd.read_csv("practice/student-data.csv")


def get_age(dob):
    now = datetime.now()
    age = relativedelta(now, dob).years
    return age


df["age"] = pd.to_datetime(df["dob"]).apply(get_age)
# print(df)

# # Splitting
# splitnames = df.copy()
# split = splitnames["name"].str.split(" ", expand=True)
# splitnames["first"] = split[0]
# splitnames["last"] = split[1]
# # print(splitnames)

# # adding a column for abb and combining
# splitnames["name_abb"] = splitnames["first"] + ", " + splitnames["last"]
# print(splitnames)

# df["target"].hist()

# Scaling Transforms

# numeric_df = df.select_dtypes(include=[np.number])
# # standardized_df = (numeric_df - numeric_df.mean()) / numeric_df.std()
# print(numeric_df)

# scaling = df.copy()
# mean_target = np.mean(scaling["target"])
# sd_target = np.std(scaling["target"])
# scaling["standardized_target"] = (scaling["target"] - mean_target) / (sd_target)
# print(scaling)

# df.drop(["dob", "name"], axis=1, inplace=True)

print(df)


distribution = df['target'].hist()
plt.title('Distribution of Target')
plt.xlabel('Target Value')
plt.ylabel('Frequency')
plt.show()


# # Cross validation
# X = df.drop("target", axis=1)
# y = df["target"]
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=42
# )
