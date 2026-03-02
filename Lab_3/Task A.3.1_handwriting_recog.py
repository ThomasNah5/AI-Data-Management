import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn import svm  # Import svm model (as taught in lab)
from sklearn.ensemble import RandomForestClassifier  # Random Forest (as taught in lab)
from sklearn.metrics import mean_squared_error, accuracy_score
import matplotlib.pyplot as plt



# Fetch MNIST dataset from OpenML
mnist = fetch_openml('mnist_784', version=1, as_frame=False)

# Get features and labels
X, y = mnist.data, mnist.target.astype(int)

print(f"Dataset shape: {X.shape}")
print(f"Number of classes: {len(np.unique(y))}")
print(f"Classes: {np.unique(y)}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=1000, train_size=6000, random_state=42
)

# Normalize the pixel values (0-255 to 0-1)
X_train = X_train / 255.0
X_test = X_test / 255.0

print(f"Training set size: {X_train.shape[0]}")
print(f"Testing set size: {X_test.shape[0]}")

# Convert to PyTorch tensors for Linear Regression
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

# Create one-hot encoded labels for 10 classes (0-9)
y_train_onehot = torch.zeros(len(y_train), 10)
y_train_onehot[torch.arange(len(y_train)), y_train] = 1
y_test_onehot = torch.zeros(len(y_test), 10)
y_test_onehot[torch.arange(len(y_test)), y_test] = 1

# Keep original labels for MSE calculation (to compare with SVM/RF)
y_train_labels = torch.tensor(y_train, dtype=torch.long)
y_test_labels = torch.tensor(y_test, dtype=torch.long)

# Training parameters
n_epochs = 10
batch_size = 64

# Store MSE history for each algorithm
mse_lr = []  # Linear Regression
mse_svm = []  # SVM
mse_rf = []  # Random Forest



model_lr = nn.Sequential(
    nn.Linear(784, 10),
    nn.Softmax(dim=1)
)
print("Model architecture:")
print(model_lr)


loss_fn = nn.MSELoss()
optimizer = optim.Adam(model_lr.parameters(), lr=0.001)

print("\nTraining Linear Regression model...")
for epoch in range(n_epochs):
    model_lr.train()
    for i in range(0, len(X_train_tensor), batch_size):
        # Take a batch
        Xbatch = X_train_tensor[i:i+batch_size]
        ybatch = y_train_onehot[i:i+batch_size]
        
        # Forward pass
        y_pred = model_lr(Xbatch)
        loss = loss_fn(y_pred, ybatch)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Update weights
        optimizer.step()
    
    
    model_lr.eval()
    with torch.no_grad():
        y_pred_test = model_lr(X_test_tensor)
        y_pred_classes = torch.argmax(y_pred_test, dim=1).numpy()
        mse = mean_squared_error(y_test, y_pred_classes)
        mse_lr.append(mse)
    
    print(f"Epoch {epoch+1}/{n_epochs} - MSE: {mse:.4f}")

# Training SVM (Linear Kernel) model..

for epoch in range(n_epochs):
  
    subset_size = int(len(X_train) * (epoch + 1) / n_epochs)
    X_subset = X_train[:subset_size]
    y_subset = y_train[:subset_size]
    
    # Create a svm Classifier with Linear Kernel 
    clf_svm = svm.SVC(kernel='linear', C=1.0)
    
    # Train the model using the training sets
    clf_svm.fit(X_subset, y_subset)
    
    # Predict the response for test dataset
    y_pred = clf_svm.predict(X_test)
    
    # Calculate MSE
    mse = mean_squared_error(y_test, y_pred)
    mse_svm.append(mse)
    
    print(f"Epoch {epoch+1}/{n_epochs} (samples={subset_size}) - MSE: {mse:.4f}")

# Simulate epochs by increasing number of estimators
max_depth = 15  
print(f"\nTraining Random Forest (max_depth={max_depth})...")

for epoch in range(n_epochs):
    # Incrementally increase estimators to simulate epochs
    n_estimators = (epoch + 1) * 10
    
    rf_clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42,
        n_jobs=-1
    )
    
 
    rf_clf.fit(X_train, y_train)
    y_pred = rf_clf.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mse_rf.append(mse)
    
    print(f"Epoch {epoch+1}/{n_epochs} (n_estimators={n_estimators}) - MSE: {mse:.4f}")



# Linear Regression accuracy
with torch.no_grad():
    y_pred_lr = model_lr(X_test_tensor)
    # Get predicted class (argmax of 10 outputs)
    y_pred_lr_classes = torch.argmax(y_pred_lr, dim=1).numpy()
lr_accuracy = accuracy_score(y_test, y_pred_lr_classes)

# SVM accuracy
y_pred_svm = clf_svm.predict(X_test)
svm_accuracy = accuracy_score(y_test, y_pred_svm)

# Random Forest accuracy
y_pred_rf = rf_clf.predict(X_test)
rf_accuracy = accuracy_score(y_test, y_pred_rf)

print(f"Linear Regression Accuracy: {lr_accuracy*100:.2f}%")
print(f"SVM (Linear Kernel) Accuracy: {svm_accuracy*100:.2f}%")
print(f"Random Forest Accuracy: {rf_accuracy*100:.2f}%")

print("\nFinal MSE Values:")
print(f"Linear Regression MSE: {mse_lr[-1]:.4f}")
print(f"SVM MSE: {mse_svm[-1]:.4f}")
print(f"Random Forest MSE: {mse_rf[-1]:.4f}")


epochs = range(1, n_epochs + 1)

plt.figure(figsize=(10, 6))

# Plot MSE for each algorithm with different colors
plt.plot(epochs, mse_svm, 'b-o', label='SVM', linewidth=2, markersize=6)
plt.plot(epochs, mse_lr, 'r-s', label='LR', linewidth=2, markersize=6)
plt.plot(epochs, mse_rf, 'g-^', label='RF', linewidth=2, markersize=6)

# Customize the plot
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Mean Squared Error (MSE)', fontsize=12)
plt.title('MSE Error vs Epoch for Handwriting Recognition (MNIST)', fontsize=14)
plt.legend(loc='upper right', fontsize=11)
plt.grid(True, alpha=0.3)
plt.xticks(epochs)

# Adjust layout and save
plt.tight_layout()
plt.savefig('Lab_3/mse_vs_epoch.png', dpi=150)
print("Plot saved as 'Lab_3/mse_vs_epoch.png'")
plt.show()
