import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, log_loss
import matplotlib.pyplot as plt




mnist = fetch_openml('mnist_784', version=1, as_frame=False)

X, y = mnist.data, mnist.target.astype(int)

print(f"Dataset shape: {X.shape}")
print(f"Number of classes: {len(np.unique(y))}")
print(f"Classes: {np.unique(y)}")


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=10000, train_size=6000, random_state=42
)


X_train = X_train / 255.0
X_test = X_test / 255.0


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Training set size: {X_train.shape[0]}")
print(f"Testing set size: {X_test.shape[0]}")


n_classes = 10
y_test_onehot = np.zeros((len(y_test), n_classes))
y_test_onehot[np.arange(len(y_test)), y_test] = 1


n_epochs = 10
batch_size = 64

# Store metrics history
mse_logreg = []      
mse_svm = []         
mse_rf = []          
acc_logreg = []      
acc_svm = []         
acc_rf = []          


def compute_mse_from_proba(y_true_onehot, y_proba):
    return np.mean((y_true_onehot - y_proba) ** 2)


def shuffle_data(X, y):
    indices = np.random.permutation(len(X))
    return X[indices], y[indices]




# SGDClassifier with log_loss 
clf_logreg = SGDClassifier(
    loss='log_loss',           
    penalty='l2',
    alpha=0.0001,
    max_iter=1,
    warm_start=True,
    random_state=42
)

classes = np.arange(10)  

for epoch in range(n_epochs):
    # Shuffle data each epoch
    X_shuffled, y_shuffled = shuffle_data(X_train_scaled, y_train)
    
    # Train with partial_fit (true epoch-based training)
    for i in range(0, len(X_shuffled), batch_size):
        X_batch = X_shuffled[i:i+batch_size]
        y_batch = y_shuffled[i:i+batch_size]
        clf_logreg.partial_fit(X_batch, y_batch, classes=classes)
    
    
    decision_scores = clf_logreg.decision_function(X_test_scaled)
 
    exp_scores = np.exp(decision_scores - np.max(decision_scores, axis=1, keepdims=True))
    y_proba = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    
    y_pred = clf_logreg.predict(X_test_scaled)
    
    # Compute MSE on probabilities 
    mse = compute_mse_from_proba(y_test_onehot, y_proba)
    acc = accuracy_score(y_test, y_pred)
    
    mse_logreg.append(mse)
    acc_logreg.append(acc)
    
    print(f"Epoch {epoch+1}/{n_epochs} - MSE: {mse:.4f} | Accuracy: {acc*100:.2f}%")




# SGDClassifier with hinge loss
clf_svm = SGDClassifier(
    loss='hinge',              
    penalty='l2',
    alpha=0.0001,
    max_iter=1,
    warm_start=True,
    random_state=42
)

for epoch in range(n_epochs):
    # Shuffle data each epoch
    X_shuffled, y_shuffled = shuffle_data(X_train_scaled, y_train)
    
    # Train with partial_fit (true epoch-based training)
    for i in range(0, len(X_shuffled), batch_size):
        X_batch = X_shuffled[i:i+batch_size]
        y_batch = y_shuffled[i:i+batch_size]
        clf_svm.partial_fit(X_batch, y_batch, classes=classes)
    

    decision_scores = clf_svm.decision_function(X_test_scaled)
    
    # Apply softmax to convert decision scores to probabilities
    exp_scores = np.exp(decision_scores - np.max(decision_scores, axis=1, keepdims=True))
    y_proba = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    
    y_pred = clf_svm.predict(X_test_scaled)
    
  
    mse = compute_mse_from_proba(y_test_onehot, y_proba)
    acc = accuracy_score(y_test, y_pred)
    
    mse_svm.append(mse)
    acc_svm.append(acc)
    
    print(f"Epoch {epoch+1}/{n_epochs} - MSE: {mse:.4f} | Accuracy: {acc*100:.2f}%")



max_depth = 15

clf_rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
clf_rf.fit(X_train, y_train)
y_proba = clf_rf.predict_proba(X_test)
y_pred = clf_rf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
mse_rf = compute_mse_from_proba(y_test_onehot, y_proba)
print(f"Random Forest - MSE: {mse_rf:.4f} | Accuracy: {acc*100:.2f}%")



epochs = range(1, n_epochs + 1)

plt.figure(figsize=(10, 6))
plt.plot(epochs, mse_svm, 'b-o', label='SVM', linewidth=2, markersize=6)
plt.plot(epochs, mse_logreg, 'r-s', label='LR', linewidth=2, markersize=6)


plt.hlines(mse_rf, epochs[0], epochs[-1], colors='g', linestyles='--', label='RF', linewidth=2)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Mean Squared Error (MSE)', fontsize=12)
plt.title('MSE Error vs Epoch for Handwriting Recognition (MNIST)', fontsize=14)
plt.legend(loc='upper right', fontsize=11)
plt.grid(True, alpha=0.3)
plt.xticks(epochs)

plt.tight_layout()
plt.savefig('Lab_3/mse_vs_epoch.png', dpi=150)
print("\nPlot saved as 'Lab_3/mse_vs_epoch.png'")
plt.show()


