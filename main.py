from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
import pydotplus
from sklearn.tree import export_graphviz
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score
import graphviz
import numpy as np

#Step 1: Preparing data sets
# Read file and load data
data = pd.read_csv("nursery.data.csv", header=None)

# Define column names
columns = ['parents', 'has_nurs', 'form', 'children', 'housing', 'finance', 'social', 'health', 'target']
data.columns = columns

print("[THE ORIGINAL SET]\n", data)
print("------------------------------------\n")

# Define ratios for train/test split
ratioList = [(0.4, 0.6), (0.6, 0.4), (0.8, 0.2), (0.9, 0.1)]

# List to store train/test splits
data_splits = []

#Spliting data 
for ratio in ratioList:
    # Splitting data 
    X_train, X_test, y_train, y_test = train_test_split(data.drop('target', axis=1), data['target'], 
                                                        train_size=ratio[0], test_size=ratio[1], 
                                                        shuffle=True, random_state=50, stratify=data['target'])
    
    # Store the splits in a dictionary
    split = {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test
    }
    
    data_splits.append(split)
    print("[Feature_train]\n",X_train)
    print("------------------------------------\n")
    print("[Label_train]\n",y_train)
    print("------------------------------------\n")
    print("[Feature_test]\n",X_test)
    print("------------------------------------\n")
    print("[Label_test]\n",y_test)
    print("------------------------------------\n")


    #Step 2: Building the decision tree classifiers
    treeClassifier = DecisionTreeClassifier(criterion='entropy', max_depth=None, random_state=50)

    # Label encoding for categorical variables
    label_encoders = {}
    X_train_encoded = X_train.copy()
    X_test_encoded = X_test.copy()
    for col in columns[:-1]:  # Exclude the target variable
        le = LabelEncoder()
        X_train_encoded[col] = le.fit_transform(X_train[col])
        X_test_encoded[col] = le.transform(X_test[col])
        label_encoders[col] = le

    treeClassifier.fit(X_train_encoded, y_train)

    # Visualize the decision tree
    dot_data = export_graphviz(treeClassifier, out_file=None, 
                            feature_names=columns[:-1],  
                            class_names=y_train.unique(),  
                            filled=True, rounded=True,  
                            special_characters=True)  

    graph = pydotplus.graph_from_dot_data(dot_data)
    graph.write_png("decision_tree.png")

    #Step 3: Evaluating the decision tree classifiers
    # Predict on the test set
    y_pred = treeClassifier.predict(X_test_encoded)

    # classification_report
    print("Classification Report:\n", classification_report(y_test, y_pred))

    # confusion_matrix.
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()

    classes = y_train.unique()
    tick_marks = range(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in [(i, j) for i in range(cm.shape[0]) for j in range(cm.shape[1])]:
        plt.text(j, i, format(cm[i, j], 'd'),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black")

    #Displaying the confusion matrix
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

#Step 4: The depth and accuracy of a decision tree (80/20)

X_train, X_test, y_train, y_test = data_splits[2].values()

# Building the decision tree classifiers for different max_depth values
max_depth_values = [None, 2, 3, 4, 5, 6, 7]
accuracy_scores = []

for max_depth in max_depth_values:
    # Building the decision tree classifiers
    treeClassifier = DecisionTreeClassifier(criterion='entropy', max_depth=max_depth, random_state=50)
    
    # Label encoding for categorical features
    label_encoders = {}
    X_train_encoded = X_train.copy()
    X_test_encoded = X_test.copy()
    for col in columns[:-1]:  # Exclude the target variable
        le = LabelEncoder()
        X_train_encoded[col] = le.fit_transform(X_train[col])
        X_test_encoded[col] = le.transform(X_test[col])
        label_encoders[col] = le

    treeClassifier.fit(X_train_encoded, y_train)
    
    # Predict on the test set
    y_pred = treeClassifier.predict(X_test_encoded)
    
    # Calculate accuracy score
    accuracy = accuracy_score(y_test, y_pred)
    accuracy_scores.append(np.float32(accuracy))
    
    # Visualize the decision tree
    dot_data = export_graphviz(treeClassifier, out_file=None, 
                               feature_names=columns[:-1],  
                               class_names=y_train.unique(),  
                               filled=True, rounded=True,  
                               special_characters=True)  

    graph = graphviz.Source(dot_data)
    graph.render(filename=f"decision_tree_max_depth_{max_depth}")

print("Accuracy Scores for Different max_depth Values:")
print("max_depth\tAccuracy")
for i, (max_depth, accuracy) in enumerate(zip(max_depth_values, accuracy_scores)):
    print(f"{max_depth}\t\t{accuracy:.4f}")
