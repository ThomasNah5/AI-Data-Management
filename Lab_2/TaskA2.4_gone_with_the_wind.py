import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


columns = ['Date Time', 'press (mbar)', 'Temp (degC)', 'Temppot (K)', 'Tempdew (degC)',
           'relativehum (%)', 'VPressmax (mbar)', 'VPressact (mbar)', 'VPressdef (mbar)',
           'sh (g/kg)', 'H2OC (mmol/mol)', 'relativeho (g/m**3)', 'windvelo (m/s)',
           'max. windvelo (m/s)', 'winddeg (deg)']


df = pd.read_csv('Lab_2/data/gone_the_wind/Climate2016.csv', skiprows=1, header=None, names=columns)


wind_velo = df['windvelo (m/s)']
wind_deg = df['winddeg (deg)']

print("Wind Velocity Data Summary:")
print(wind_velo.describe())
print("Wind Direction Data Summary:")
print(wind_deg.describe())


# Plot hist2d of wind direction vs wind velocity before
plt.figure(figsize=(8, 6))
plt.hist2d(wind_deg, wind_velo, bins=(50, 50), vmax=400, cmap='viridis')
plt.colorbar()
plt.xlabel('Wind Direction [deg]')
plt.ylabel('Wind Velocity [m/s]')
plt.title('Wind Data - Before Transformation')
plt.savefig('Lab_2/data/gone_the_wind/before_transformation.png', dpi=150, bbox_inches='tight')
plt.show()


# Convert wind direction from degrees to radians
wind_rad = np.deg2rad(wind_deg)


# Calculate X and Y components of wind vector
windveloX = wind_velo * np.cos(wind_rad)
windveloY = wind_velo * np.sin(wind_rad)


# Using the normalize function to normalize the data
def normalize(data):
    return (data - data.min()) / (data.max() - data.min())

# Normalize the X and Y vectors
windveloX_normalized = normalize(windveloX)
windveloY_normalized = normalize(windveloY)

print("Normalized Wind Vector X Statistics:")
print(windveloX_normalized.describe())
print("Normalized Wind Vector Y Statistics:")
print(windveloY_normalized.describe())


# Add new columns to the dataframe
df['windveloX'] = windveloX
df['windveloY'] = windveloY


df.to_csv('Lab_2/data/gone_the_wind/Climate2016.csv', index=False)


# Plot hist2d of wind vector X vs Y after
plt.figure(figsize=(8, 6))
plt.hist2d(windveloX, windveloY, bins=(50, 50), vmax=400, cmap='viridis')
plt.colorbar()
plt.xlabel('Wind Velocity X [m/s]')
plt.ylabel('Wind Velocity Y [m/s]')
plt.title('After Transformation (X-Y Vector)')
plt.savefig('Lab_2/data/gone_the_wind/after_transformation.png', dpi=150, bbox_inches='tight')
plt.show()

