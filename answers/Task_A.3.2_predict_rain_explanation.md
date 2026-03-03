# Task A.3.2: Predict Rain (Seattle Weather) - Code Explanation

## Overview

This task uses the **Seattle Weather dataset** to predict weather conditions based on features like precipitation, temperature, and wind. This is a **multiclass classification** problem with 5 weather categories.

---

## PART 1: Problem Understanding

### Dataset Information

- **Source**: `seattle-weather.csv`
- **Features**: precipitation, temp_max, temp_min, wind
- **Target**: weather (categorical: drizzle, fog, rain, snow, sun)
- **Task Type**: **Multiclass Classification** (5 classes)

### Weather Categories

| Label (Encoded) | Weather Type |
| --------------- | ------------ |
| 0               | drizzle      |
| 1               | fog          |
| 2               | rain         |
| 3               | snow         |
| 4               | sun          |

---

## PART 2: Line-by-Line Code Explanation

### Section 1: Imports (Lines 1-9)

```python
import pandas as pd                          # Data manipulation
import numpy as np                           # Numerical operations
import matplotlib.pyplot as plt              # Plotting
from sklearn.model_selection import train_test_split  # Split data
from sklearn.preprocessing import LabelEncoder, StandardScaler  # Preprocessing
from sklearn.linear_model import SGDClassifier       # For LR and SVM
from sklearn.svm import SVC                          # Alternative SVM (not used here)
from sklearn.ensemble import RandomForestClassifier  # Random Forest
from sklearn.metrics import mean_squared_error, confusion_matrix, ConfusionMatrixDisplay  # Evaluation
```

**To Professor:** "I imported scikit-learn's SGDClassifier for epoch-based training, along with preprocessing tools and evaluation metrics."

---

### Section 2: Load and Explore Data (Lines 12-18)

```python
df = pd.read_csv("Lab_3/seattle-weather.csv")

print("Dataset Shape:", df.shape)
print("First 5 rows:")
print(df.head())
print("\nWeather categories:", df['weather'].unique())
```

- **`pd.read_csv()`**: Loads the CSV file into a pandas DataFrame
- **`df.shape`**: Shows dimensions (rows × columns)
- **`df['weather'].unique()`**: Lists all unique weather categories

**To Professor:** "I loaded the Seattle weather dataset and explored its structure to understand the features and target variable."

---

### Section 3: Prepare Features and Target (Lines 21-22)

```python
X = df[['precipitation', 'temp_max', 'temp_min', 'wind']].values
y = df['weather'].values
```

- **`X`**: Feature matrix with 4 columns (precipitation, max temp, min temp, wind)
- **`y`**: Target variable (weather category as strings)

**To Professor:** "I extracted the four numerical features as input (X) and the weather category as the target (y)."

---

### Section 4: Label Encoding (Lines 25-27)

```python
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
print("\nLabel mapping:", dict(zip(label_encoder.classes_, range(len(label_encoder.classes_)))))
```

**Why Label Encoding?**

- Machine learning algorithms require numerical inputs
- Converts: `['drizzle', 'fog', 'rain', 'snow', 'sun']` → `[0, 1, 2, 3, 4]`
- **`fit_transform()`**: Learns the mapping and applies it in one step

**To Professor:** "I used LabelEncoder to convert categorical weather labels into numerical values that the classifiers can process."

---

### Section 5: Feature Scaling (Lines 30-31)

```python
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

**StandardScaler** transforms each feature to have:

- **Mean = 0**
- **Standard deviation = 1**

**Why scale?**

- SGDClassifier uses gradient descent → converges faster with scaled features
- Prevents features with larger magnitudes (e.g., temperature) from dominating

**Formula:** z = (x - μ) / σ

**To Professor:** "I standardized the features to zero mean and unit variance, which improves gradient descent convergence for the SGDClassifier."

---

### Section 6: Train/Test Split (Lines 34-37)

```python
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.2, random_state=42
)
```

- **80% training, 20% testing**
- **`random_state=42`**: Ensures reproducibility

**To Professor:** "I split the data into 80% training and 20% testing sets with a fixed random state for reproducibility."

---

### Section 7: Logistic Regression Training (Lines 47-55)

```python
lr_model = SGDClassifier(loss='log_loss', max_iter=1, warm_start=True, random_state=42)

for epoch in range(n_epochs):
    lr_model.partial_fit(X_train, y_train, classes=np.unique(y_encoded))
    y_pred_lr = lr_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred_lr)
    mse_lr.append(mse)
```

**Key Parameters:**

| Parameter         | Value                    | Meaning                                     |
| ----------------- | ------------------------ | ------------------------------------------- |
| `loss='log_loss'` | Log loss (cross-entropy) | **This makes it Logistic Regression**       |
| `warm_start=True` | Retain weights           | Continues training from previous epoch      |
| `max_iter=1`      | One iteration per call   | Combined with partial_fit for epoch control |

**`partial_fit()`**: Performs one epoch of training, enabling us to track MSE after each epoch.

**To Professor:** "I used SGDClassifier with log_loss, which implements multinomial logistic regression. The partial_fit method allows epoch-by-epoch training to track the learning curve."

---

### Section 8: SVM Training (Lines 59-67)

```python
svm_model = SGDClassifier(loss='hinge', max_iter=1, warm_start=True, random_state=42)

for epoch in range(n_epochs):
    svm_model.partial_fit(X_train, y_train, classes=np.unique(y_encoded))
    y_pred_svm = svm_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred_svm)
    mse_svm.append(mse)
```

**Key Difference from Logistic Regression:**

| Loss Function | Algorithm              |
| ------------- | ---------------------- |
| `log_loss`    | Logistic Regression    |
| `hinge`       | Support Vector Machine |

**Hinge Loss:** L = max(0, 1 - y × f(x))

- This is the standard SVM loss function
- SGDClassifier with hinge loss = Linear SVM optimized via Stochastic Gradient Descent

**To Professor:** "I used SGDClassifier with hinge loss, which is mathematically equivalent to a linear SVM. This allows epoch-based training while maintaining the SVM decision boundary."

---

### Section 9: Random Forest Training (Lines 70-78)

```python
for epoch in range(n_epochs):
    n_estimators = epoch + 1  # 1, 2, 3, ..., 50 trees
    rf_model = RandomForestClassifier(n_estimators=n_estimators, max_depth=9, random_state=42)
    rf_model.fit(X_train, y_train)
    y_pred_rf = rf_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred_rf)
    mse_rf.append(mse)
```

**Key Parameters:**

- **`max_depth=9`**: Maximum tree depth (< 10 as required)
- **`n_estimators`**: Number of trees (simulates epochs)

**Simulating Epochs:**

- Random Forest doesn't have epochs (it's an ensemble method)
- We simulate epochs by **incrementally increasing the number of trees**
- Epoch 1 → 1 tree, Epoch 2 → 2 trees, ..., Epoch 50 → 50 trees

**To Professor:** "Random Forest is an ensemble method without inherent epochs. I simulated the learning progression by incrementally increasing the number of trees from 1 to 50, demonstrating how the ensemble improves with more estimators."

---

### Section 10: MSE vs Epoch Plot (Lines 89-103)

```python
plt.figure(figsize=(10, 6))
epochs = range(1, n_epochs + 1)

plt.plot(epochs, mse_lr, 'b-', label='LR', linewidth=2)
plt.plot(epochs, mse_svm, 'r-', label='SVM', linewidth=2)
plt.plot(epochs, mse_rf, 'g-', label='RF', linewidth=2)

plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Mean Squared Error (MSE)', fontsize=12)
plt.title('MSE vs Epoch for Three Classification Algorithms', fontsize=14)
plt.legend(loc='upper right', fontsize=11)
```

**Plot Details:**

- **Blue line**: Logistic Regression (LR)
- **Red line**: SVM
- **Green line**: Random Forest (RF)
- **Legend**: Upper right corner with labels "LR", "SVM", "RF"

**To Professor:** "I created a line plot showing MSE progression across epochs for all three algorithms, with distinct colors and a legend in the upper right corner."

---

### Section 11: Confusion Matrix (Lines 107-115)

```python
cm = confusion_matrix(y_test, y_pred_rf_final)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(cmap='Blues', values_format='d')
plt.title('Confusion Matrix - Random Forest Classifier', fontsize=14)
```

**What is a Confusion Matrix?**

- A table showing predicted vs actual classes
- Diagonal elements = correct predictions
- Off-diagonal elements = misclassifications

**Example interpretation:**

```
              Predicted
           drizzle  fog  rain  snow  sun
Actual
drizzle       15     2    5     0    1
fog            3    12    2     0    0
rain           8     1   45     0    3
...
```

**To Professor:** "I generated a confusion matrix using the Random Forest predictions to visualize classification performance across all weather categories. The diagonal shows correct predictions, while off-diagonal cells show misclassifications."

---

## PART 3: Algorithm Comparison Summary

| Algorithm                    | Description                              | Epoch Tracking Method           |
| ---------------------------- | ---------------------------------------- | ------------------------------- |
| **Logistic Regression (LR)** | Multinomial classification using softmax | `partial_fit()` with log_loss   |
| **SVM (Linear)**             | Support Vector Machine with hinge loss   | `partial_fit()` with hinge loss |
| **Random Forest (RF)**       | Ensemble of decision trees (max_depth=9) | Incrementing `n_estimators`     |

---

## PART 4: How to Explain to Your Professor

### Opening Statement

> "This task predicts weather conditions from the Seattle Weather dataset. I used three classification algorithms: Logistic Regression, Linear SVM, and Random Forest with max_depth=9."

### On Algorithm Choice

> "I used SGDClassifier for both Logistic Regression (log_loss) and SVM (hinge loss) because it enables epoch-based training through partial_fit(), allowing us to track MSE progression over time."

### On Random Forest Epochs

> "Since Random Forest doesn't have epochs, I simulated the learning process by incrementally adding trees to the ensemble. This shows how the model improves as the ensemble grows."

### On MSE for Classification

> "MSE is computed by comparing predicted class labels with true labels. While accuracy is the standard classification metric, MSE provides a continuous error measure useful for visualizing training dynamics."

### On the Confusion Matrix

> "The confusion matrix provides a detailed breakdown of classification performance, showing which weather types are correctly predicted and where the model makes errors."

---

## PART 5: Key Concepts Summary

### 1. Why StandardScaler?

Gradient-based methods (SGDClassifier) converge faster when features are on the same scale. Without scaling, features with larger values dominate the learning process.

### 2. Why LabelEncoder?

Machine learning models need numerical inputs. LabelEncoder converts categorical strings to integers while preserving the mapping for interpretation.

### 3. Why partial_fit()?

Allows incremental training one epoch at a time, enabling us to:

- Track MSE after each epoch
- Visualize the learning curve
- Compare convergence speed between algorithms

### 4. Why max_depth < 10 for Random Forest?

- Limits tree complexity to prevent overfitting
- Ensures trees don't memorize training data
- Balances bias-variance tradeoff

---

## Expected Results

| Algorithm           | Expected Behavior                                  |
| ------------------- | -------------------------------------------------- |
| Logistic Regression | MSE decreases rapidly then stabilizes              |
| SVM                 | Similar to LR, may have different convergence rate |
| Random Forest       | MSE decreases as more trees are added              |

The confusion matrix will show:

- **Sun** and **Rain** are typically well-predicted (more samples)
- **Snow** may have few predictions (rare class)
- **Drizzle** may be confused with **Rain** (similar conditions)

---

## Output Files

- **`Lab_3/mse_epoch_plot.png`**: MSE vs Epoch for all three algorithms
- **`Lab_3/confusion_matrix.png`**: Confusion matrix for Random Forest
