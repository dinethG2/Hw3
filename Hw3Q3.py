from math import e
import pandas as pd  
import numpy as np  
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold

# Sigmoid function
def sig(val):
    return 1/(1+np.exp(-val))

    
# Read data into dataframe
df = pd.read_csv("data/emails.csv")    

# Split into 5 folds for cross validation
alpha = 0.01
kf = KFold(n_splits=5)
fold = 0
for train, test in kf.split(df):

    #  Split data and labels for testing and training
    X_train = df.iloc[np.array(train)].drop(columns=['Prediction', "Email No."])
    y_train = df.iloc[np.array(train)]['Prediction'].values
    X_test = df.iloc[np.array(test)].drop(columns=['Prediction', "Email No."])
    y_test = df.iloc[np.array(test)]['Prediction'].values

    # Set value of theta
    theta = np.zeros(3001)

    # Prepare training data
    bias = np.ones(4000)
    X_train["Bias"] = bias
    cols = X_train.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    X_train = X_train[cols]
    X = np.transpose(X_train.values)

    # Gradient descent
    for i in range(1000):

        Z = np.matmul(theta,X)
        A = sig(Z)
        dZ = A - y_train
        dtheta = (1/4000) * np.matmul(X,np.transpose(dZ))
        
        # Update the value of theta
        theta = theta - alpha * dtheta
        

    # Prepare test data
    bias = np.ones(1000)
    X_test["Bias"] = bias
    cols = X_test.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    X_test = X_test[cols]

    # Predict test data
    y_pred = sig(np.matmul(X_test, theta))
    y_pred = np.array(y_pred)
   
    # Calculate labels based on probabilities
    for i in range(len(y_pred)):
        if(y_pred[i]>=0.5):
            y_pred[i] = 1
        else:
            y_pred[i] = 0


    #  Calculate Accuracy, Precision and recall of each fold
    error = abs(y_pred - y_test).sum()
    accuracy = (len(y_pred) - error)/len(y_pred)
    precision = metrics.precision_score(y_test, y_pred)
    recall = metrics.recall_score(y_test, y_pred)


    # Print Accuracy, Precision an recall of each fold
    fold+=1
    print("Fold: ", fold,", Test:", test[0], "...", test[len(test)-1])    
    print("Accuracy: ", accuracy)
    print("Precision: ", precision)
    print("Recall: ", recall)
    print()


