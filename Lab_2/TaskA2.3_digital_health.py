import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


df = pd.read_csv('Lab_2/data/digital_health/aw_fb_data.csv')


# Normalizing distribution of calories column

fig, ax = plt.subplots(figsize=(10, 6))
fig.suptitle('Calories Cubic Transformation', fontsize=14, fontweight='bold')


# Cube transformation
calories_cube = np.cbrt(df['calories'])
ax.hist(calories_cube, bins=50, color='purple', edgecolor='black', alpha=0.7)
ax.set_title('Cubic Transformation')
ax.set_xlabel('Calories')
ax.set_ylabel('Frequency')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('Lab_2/data/task1_calories_transformation.png', dpi=150, bbox_inches='tight')
plt.close()



# Making a copy of the original dataframe
df_copy = df.copy()

df_participants = df_copy.drop_duplicates(subset=['age', 'gender', 'height', 'weight'], keep='first').reset_index(drop=True)
print(df_participants)


fig, axs = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
fig.suptitle('Participant Demographics (49 Participants)', fontsize=14, fontweight='bold')


participants = range(1, len(df_participants) + 1)

# Age
axs[0].plot(participants, df_participants['age'], color='pink', markersize=4, label='Age')
axs[0].set_ylabel('Age')
axs[0].grid(True, alpha=0.3)
axs[0].legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=1, frameon=True)

# Height 
axs[1].plot(participants, df_participants['height'], color='orange', markersize=4, label='Height')
axs[1].set_ylabel('Height')
axs[1].grid(True, alpha=0.3)
axs[1].legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=1, frameon=True)

# Weight
axs[2].plot(participants, df_participants['weight'], color='red', markersize=4, label='Weight')
axs[2].set_ylabel('Weight',)
axs[2].set_xlabel('Participant ID', fontsize=11)
axs[2].grid(True, alpha=0.3)
axs[2].legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=1, frameon=True)

plt.tight_layout()
plt.savefig('Lab_2/data/task2_participant_demographics.png', dpi=150, bbox_inches='tight')
plt.close()





first_three_participants = df_copy.drop_duplicates(subset=['age', 'gender', 'height', 'weight'])[['age', 'gender', 'height', 'weight']].values[:3]

# Extract data for the first 3 participants
participant_data = []
for i, (age, gender, height, weight) in enumerate(first_three_participants):
    mask = (df['age'] == age) & (df['gender'] == gender) & (df['height'] == height) & (df['weight'] == weight)
    participant_data.append(df[mask].reset_index(drop=True))
    print(f"Participant #{i+1}: Age={age}, Height={height}, Weight={weight}, Samples={mask.sum()}")


# Create stacked subplots for steps, heart_rate, calories
fig, axs = plt.subplots(3, 1, figsize=(14, 12), sharex=False)
fig.suptitle('Activity Metrics for First 3 Participants', fontsize=14, fontweight='bold')

# Define colors for each participant
colors = ['red', 'blue', 'green']

# Steps subplot
for i, data in enumerate(participant_data):
    time_points = range(len(data))
    axs[0].plot(time_points, data['steps'], color=colors[i], linewidth=1.5, 
                 alpha=0.8, label=f'Participant #{i+1}')
axs[0].set_ylabel('Steps', fontsize=11)
axs[0].set_title('Steps Over Time', fontsize=12)
axs[0].grid(True, alpha=0.3)
axs[0].legend(loc='upper right', frameon=True, fontsize=9)

# Heart Rate subplot (note: column is 'hear_rate' in dataset - typo in original data)
for i, data in enumerate(participant_data):
    time_points = range(len(data))
    axs[1].plot(time_points, data['hear_rate'], color=colors[i], linewidth=1.5, 
                 alpha=0.8, label=f'Participant #{i+1}')
axs[1].set_ylabel('Heart Rate (bpm)', fontsize=11)
axs[1].set_title('Heart Rate Over Time', fontsize=12)
axs[1].grid(True, alpha=0.3)
axs[1].legend(loc='upper right', frameon=True, fontsize=9)

# Calories subplot
for i, data in enumerate(participant_data):
    time_points = range(len(data))
    axs[2].plot(time_points, data['calories'], color=colors[i], linewidth=1.5, 
                 alpha=0.8, label=f'Participant #{i+1}')
axs[2].set_ylabel('Calories', fontsize=11)
axs[2].set_xlabel('Time Points (samples)', fontsize=11)
axs[2].set_title('Calories Over Time', fontsize=12)
axs[2].grid(True, alpha=0.3)
axs[2].legend(loc='upper right', frameon=True, fontsize=9)

plt.tight_layout()
plt.savefig('Lab_2/data/task3_first3_participants.png', dpi=150, bbox_inches='tight')
plt.close()




# Normalization of age, height, weight and Standardization of steps, hear_rate

# Normalization of age, height, weight
df['age_normalized'] = (df['age'] - df['age'].min()) / (df['age'].max() - df['age'].min())
df['height_normalized'] = (df['height'] - df['height'].min()) / (df['height'].max() - df['height'].min())
df['weight_normalized'] = (df['weight'] - df['weight'].min()) / (df['weight'].max() - df['weight'].min())

# Standardization of steps and heart_rate 
df['steps_standardized'] = (df['steps'] - df['steps'].mean()) / df['steps'].std()
df['hear_rate_standardized'] = (df['hear_rate'] - df['hear_rate'].mean()) / df['hear_rate'].std()

# Display sample of normalized/standardized data
print("\nSample of normalized/standardized values:")
print(df[['age', 'age_normalized', 'height', 'height_normalized', 
          'weight', 'weight_normalized', 'steps', 'steps_standardized', 
          'hear_rate', 'hear_rate_standardized']].head(10))


# Split Dataset into Train (70%), Validation (15%), Test (15%)


# # First split
df_train, df_temp = train_test_split(df, test_size=0.30, random_state=42)

# Second split: 50% of remaining (15% of total) for validation, 50% for test
df_validation, df_test = train_test_split(df_temp, test_size=0.50, random_state=42)

print(f"Dataset split completed:")
print(f"Total samples: {len(df)}")
print(f"Training set: {len(df_train)} samples ({len(df_train)/len(df)*100:.1f}%)")
print(f"Validation set: {len(df_validation)} samples ({len(df_validation)/len(df)*100:.1f}%)")
print(f"Test set: {len(df_test)} samples ({len(df_test)/len(df)*100:.1f}%)")

# Verify the split
total_split = len(df_train) + len(df_validation) + len(df_test)
print(f"\n  Verification: {len(df_train)} + {len(df_validation)} + {len(df_test)} = {total_split} (Original: {len(df)})")

# Save the splits to CSV files (optional but useful)
df_train.to_csv('Lab_2/data/digital_health/train_data.csv', index=False)
df_validation.to_csv('Lab_2/data/digital_health/validation_data.csv', index=False)
df_test.to_csv('Lab_2/data/digital_health/test_data.csv', index=False)
