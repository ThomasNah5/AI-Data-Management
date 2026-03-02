import os

import matplotlib.pyplot as plt
import pandas as pd

script_dir = os.path.dirname(os.path.abspath(__file__))

csv_path = os.path.join(script_dir, "IOT-temp.csv")
plot_path = os.path.join(script_dir, "temperature_plot.png")
modified_csv_path = os.path.join(script_dir, "modified_IOT-temp.csv")


# Load the dataset
df = pd.read_csv(csv_path)


# Renaming the columns
df.rename(columns={"out/in": "out_in"}, inplace=True)


df["noted_date"] = pd.to_datetime(df["noted_date"], format="%d-%m-%Y %H:%M")

# Filter data
start_date = pd.to_datetime("2018-12-02")
end_date = pd.to_datetime("2018-12-09")  # Go to the next day to include all of 8th
week_data = df[(df["noted_date"] >= start_date) & (df["noted_date"] < end_date)]

# Separaing indoor and outdoor data
indoor_data = week_data[week_data["out_in"] == "In"]
outdoor_data = week_data[week_data["out_in"] == "Out"]


plt.figure(figsize=(15, 7))
plt.plot(
    indoor_data["noted_date"], indoor_data["temp"], "b-", label="Indoor Temperature"
)
plt.plot(
    outdoor_data["noted_date"], outdoor_data["temp"], "r-", label="Outdoor Temperature"
)

plt.title("Indoor and Outdoor Temperature (02-12-2018 to 08-12-2018)")
plt.xlabel("Date")
plt.ylabel("Temperature (°C)")
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(plot_path)
print(f"Temperature plot saved to '{plot_path}'")


df_modified = df.copy()

# Change "In" and "Out" to 1 and 0
df_modified["out_in"] = df_modified["out_in"].map({"In": 1, "Out": 0})
df_modified["date"] = df_modified["noted_date"].dt.date
df_modified["time"] = df_modified["noted_date"].dt.time


last_day_data = df_modified[df_modified["date"] == pd.to_datetime("2018-12-08").date()]
last_day_data.to_csv(modified_csv_path, index=False)
print(f"Modified data for 08-12-2018 saved to '{modified_csv_path}'")
