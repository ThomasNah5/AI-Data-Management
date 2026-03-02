# Supervised Learning
# type of machine learning where the model is trained on labeled data. map input with output.


# 5 sessions of supervised learning
# 1. Data collection: gather and prepare the dataset for training.
# 2. Data preprocessing: clean and transform the data to make it suitable for training.
# 3. Model selection: choose an appropriate algorithm or model architecture for the task.
# 4. Training: feed the preprocessed data into the model and adjust its parameters to minimize the error.
# 5. Evaluation: assess the performance of the trained model using metrics and validation techniques.
# Optional: Visualization: visualize the results and insights from the trained model to better understand its performance and behavior.


# Decision Trees
# non-parametric supervised learning method used for classification and regession.
# goal is to create a model that predicts the value of a target variable by learning simple decision rules 
# inferred from the data features.

# simple to understand and to interpret
# trees can be visualized
# requires little data preprocessing

# Disadvantages:

# decision tree learners can create over-complex trees that do not generalize the data well.
# Overfitting - when a model learns the training data too well, including noise and outliers, leading to poor performance on new, unseen data.
# decision trees can be unstable because small variations in the data might result in a completely different tree being generated. this problem is 
# mitigated by using decision trees within an ensemble.



import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import seaborn as sns
from sklearn.metrics import classification_report


dframe = pd.read_csv('practice/machine_learning/framingham.csv')

df = dframe.copy() 
df.head()

# Data preprocessing
df.drop(["education"], axis=1, inplace=True) 
df.rename(columns={'TenYearCHD': 'target'}, inplace=True)

print(df.shape) 
df.info()

df.isnull().sum()

# Handling missing data: here, filling all null values in glucose attribute with the mean value
df['glucose'].fillna(df['glucose'].mean())
df.dropna()

catag = [i for i in df.columns if len(df[i].unique()) < 4]
random = [i for i in df.columns if len(df[i].unique()) >= 4]

catag.remove('target')
print(catag)
print(random)

plt.figure(figsize=(25,20))
for n,column in enumerate(catag):
    plt.subplot(3,2,n+1)
    sns.countplot(x=df[column],hue=df["target"],data=df)
    plt.xlabel(column,fontsize=20)
    plt.ylabel("count",fontsize=20)
    plt.title(f'{column.title()}',weight='bold',fontsize=30)
    plt.tight_layout()
    
plt.show()    
    
x = df.drop('target', axis=1)
y = df['target']
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.22, random_state=42)
xtrain.shape, ytrain.shape, xtest.shape, ytest.shape
print()



# Using Decision Tree Classifier

dtree = DecisionTreeClassifier(criterion="entropy", max_depth=3)


#test for validation dataset
dtree.fit(xtrain,ytrain)
ypred_dtree=dtree.predict(xtest)
dtree_model=accuracy_score(ytest,ypred_dtree)
#test for train data
ypred_dtree2=dtree.predict(xtrain)


#Evaluation of the model
dtree_model2=accuracy_score(ytrain,ypred_dtree2)
print(f"accuracy for validation set :{dtree_model}\naccuracy for train set :{dtree_model2}")
score2 = cross_val_score(dtree, xtrain, ytrain, cv = 30)
print(f"\nafter cross validation the accuracy is {round(score2.mean(),2)}")


#Precesion, F1-score, Accuracy
print(classification_report(ytest,ypred_dtree)) 


#Visualizing the decision tree in boxes
fig = plt.figure(figsize=(25,20))
feature_names = df.columns.tolist() #saving the features in a list
feature_names.remove("target") #Removing the target (0,1) from the features
class_names = ["0", "1"] #Manually adding the target classes, 0 means patient does not have diabetes, 1 means they have
print (feature_names, class_names)
_ = tree.plot_tree(dtree, feature_names= feature_names, class_names=class_names)