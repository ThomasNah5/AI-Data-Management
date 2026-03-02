import matplotlib.pyplot as plt
import pandas as pd

# II- Visualize the data with a line graph


def visualize_data(data, title):
    """
    Visualizes the 6-axis accelerometer and gyroscope data.
    """
    plt.figure(figsize=(14, 7))

    # Plotting all 6 axes
    plt.plot(data.index, data["Ax"], label="Ax")
    plt.plot(data.index, data["Ay"], label="Ay")
    plt.plot(data.index, data["Az"], label="Az")
    plt.plot(data.index, data["Gx"], label="Gx")
    plt.plot(data.index, data["Gy"], label="Gy")
    plt.plot(data.index, data["Gz"], label="Gz")

    # Adding labels as per instructions
    # The prompt asks for time in seconds, but the data doesn't have a clear uniform time column.
    # We will use the index as a proxy for time steps.
    plt.xlabel("Time(seconds)")
    plt.ylabel("Acceleration (m.sq/s.sq)")

    # Adding grid
    plt.grid(True)

    # Adding legend
    plt.legend(loc="upper right")

    # Adding a title
    plt.title(title)

    plt.show()


# Load the data
try:
    df = pd.read_csv("accelero_gyro.csv")
except FileNotFoundError:
    print(
        "Error: 'accelero_gyro.csv' not found. Make sure the file is in the same directory."
    )
    exit()

# The 'Time' column is not a unique timestamp for each row.
# To create a time axis, we can calculate elapsed time assuming a constant sampling rate.
# A common sampling rate for such sensors is 100Hz.
SAMPLING_RATE_HZ = 100
time_seconds = [i / SAMPLING_RATE_HZ for i in range(len(df))]
df.index = time_seconds  # Set the index to be the time in seconds

# 1. Visualize the original data
visualize_data(df, "Raw Accelerometer and Gyroscope Data")

# Delete rows where acceleration is close to 0 to compress the signal
# "Close to 0" is subjective. For accelerometer data, when the device is stationary,
# Ax and Ay are close to 0, and Az is close to +/-1g (due to gravity).
# The gyroscope (Gx, Gy, Gz) values are close to 0 when not rotating.
# We will define a threshold based on the magnitude of acceleration, considering gravity on Az.
# A simple threshold approach: if the absolute values of Ax and Ay are small, and Az is near its mean (which is mostly gravity),
# then the device is likely stationary.
ax_threshold = 0.15
ay_threshold = 0.15
az_deviation_threshold = 0.15  # Deviation from the mean of Az

# We identify rows that are "stationary"
is_stationary = (
    (abs(df["Ax"]) < ax_threshold)
    & (abs(df["Ay"]) < ay_threshold)
    & (abs(df["Az"] - df["Az"].mean()) < az_deviation_threshold)
)

# We keep the rows that are NOT stationary
compressed_df = df[~is_stationary]

print(f"Original data points: {len(df)}")
print(f"Compressed data points: {len(compressed_df)}")
print(f"Removed {len(df) - len(compressed_df)} stationary data points.")

# 2. Visualize the compressed data
visualize_data(
    compressed_df,
    "Compressed Accelerometer and Gyroscope Data (Stationary points removed)",
)
