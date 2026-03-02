import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error, confusion_matrix, ConfusionMatrixDisplay

# Load the dataset
df = pd.read_csv("Lab_3/seattle-weather.csv")

# Check the data
print("Dataset Shape:", df.shape)
print("First 5 rows:")
print(df.head())
print("\nWeather categories:", df['weather'].unique())

# Prepare features and target
X = df[['precipitation', 'temp_max', 'temp_min', 'wind']].values
y = df['weather'].values


label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
print("\nLabel mapping:", dict(zip(label_encoder.classes_, range(len(label_encoder.classes_)))))

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)

print(f"\nTraining set size: {len(X_train)}")
print(f"Testing set size: {len(X_test)}")


# Define number of epochs for tracking MSE
n_epochs = 50

# Store MSE values for each epoch
mse_lr = []  
mse_svm = []  
mse_rf = []  


lr_model = SGDClassifier(loss='log_loss', max_iter=1, warm_start=True, random_state=42)

for epoch in range(n_epochs):
    lr_model.partial_fit(X_train, y_train, classes=np.unique(y_encoded))
    y_pred_lr = lr_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred_lr)
    mse_lr.append(mse)
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch + 1}: MSE = {mse:.4f}")



svm_model = SGDClassifier(loss='hinge', max_iter=1, warm_start=True, random_state=42)

for epoch in range(n_epochs):
    svm_model.partial_fit(X_train, y_train, classes=np.unique(y_encoded))
    y_pred_svm = svm_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred_svm)
    mse_svm.append(mse)
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch + 1}: MSE = {mse:.4f}")

# Random Forest (max_depth < 10)

for epoch in range(n_epochs):
    n_estimators = epoch + 1
    rf_model = RandomForestClassifier(n_estimators=n_estimators, max_depth=9, random_state=42)
    rf_model.fit(X_train, y_train)
    y_pred_rf = rf_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred_rf)
    mse_rf.append(mse)
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch + 1} (n_trees={n_estimators}): MSE = {mse:.4f}")

# Final predictions
y_pred_lr_final = lr_model.predict(X_test)
y_pred_svm_final = svm_model.predict(X_test)
y_pred_rf_final = rf_model.predict(X_test)

print("\n=== Final Results ===")
print(f"Linear Regression Final MSE: {mse_lr[-1]:.4f}")
print(f"SVM (Linear) Final MSE: {mse_svm[-1]:.4f}")
print(f"Random Forest Final MSE: {mse_rf[-1]:.4f}")

# Visualizing MSE error against Epoch
plt.figure(figsize=(10, 6))
epochs = range(1, n_epochs + 1)

plt.plot(epochs, mse_lr, 'b-', label='LR', linewidth=2)
plt.plot(epochs, mse_svm, 'r-', label='SVM', linewidth=2)
plt.plot(epochs, mse_rf, 'g-', label='RF', linewidth=2)

plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Mean Squared Error (MSE)', fontsize=12)
plt.title('MSE vs Epoch for Three Classification Algorithms', fontsize=14)
plt.legend(loc='upper right', fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('Lab_3/mse_epoch_plot.png', dpi=150)
plt.show()


# Confusion Matrix (Using Random Forest)

plt.figure(figsize=(8, 6))
class_names = label_encoder.classes_
cm = confusion_matrix(y_test, y_pred_rf_final)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(cmap='Blues', values_format='d')
plt.title('Confusion Matrix - Random Forest Classifier', fontsize=14)
plt.tight_layout()
plt.savefig('Lab_3/confusion_matrix.png', dpi=150)
plt.show()


