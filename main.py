from sklearn.model_selection import*
from sklearn.preprocessing import *
import pandas as pd
import numpy as np

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
print("[feature_train]\n",X_train)
print("------------------------------------\n")
print("[Label_train]\n",y_train)
print("------------------------------------\n")
print("[Feature_test]\n",X_test)
print("------------------------------------\n")
print("[Label_test]\n",y_test)
print("------------------------------------\n")

#Building the decision tree classifiers