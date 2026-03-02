import matplotlib.pyplot as plt
import pandas as pd

# import numpy as np

df = pd.read_csv("temperature.csv")

timesteps = df["Timesteps"]
temperatures = df["Temperature"]

plt.figure(figsize=(12, 6))
plt.plot(timesteps, temperatures, color="orange", label="Temperature")
plt.xlabel("Time (seconds)")
plt.ylabel("Temperature (degrees Celsius)")


plt.title("Arduino Temperature Sensor Data")
plt.grid(True)
plt.legend(loc="upper right")
plt.show()
