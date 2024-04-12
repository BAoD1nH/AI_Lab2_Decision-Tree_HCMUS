from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
import pydotplus
from sklearn.tree import export_graphviz
import matplotlib.pyplot as plt


#Step 1: Prepare data sets
# Read file and load data
data = pd.read_csv("nursery.data.csv", header=None)

# Define column names
columns = ['parents', 'has_nurs', 'form', 'children', 'housing', 'finance', 'social', 'health', 'target']
data.columns = columns

print("[THE ORIGINAL SET]\n", data)
print("------------------------------------\n")

#Spliting data 
#(train/test) = 40/60
X_train, X_test, y_train, y_test = train_test_split(data.drop('target', axis=1), data['target'], 
                                                    train_size= 0.4, test_size=0.6, shuffle=True,
                                                      random_state=50, stratify=data['target'])
print("[Feature_train]\n",X_train)
print("------------------------------------\n")
print("[Label_train]\n",y_train)
print("------------------------------------\n")
print("[Feature_test]\n",X_test)
print("------------------------------------\n")
print("[Label_test]\n",y_test)
print("------------------------------------\n")

#Building the decision tree classifiers
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

# Display the decision tree visualization
plt.figure(figsize=(20,20))
plt.imshow(plt.imread("decision_tree.png"))
plt.axis('off')
plt.show()