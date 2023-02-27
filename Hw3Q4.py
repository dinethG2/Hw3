from matplotlib import pyplot as plt
import pandas as pd  
import numpy as np  
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold

# Question 2
# making dataframe 
df = pd.read_csv("data/emails.csv") 

# Run kNN with different K values   
kVals = [1,3,5,7,10]
avg_acc = []
for k in kVals:

    # Split into 5 folds for cross validation
    kf = KFold(n_splits=5)
    accuracy = []
    for train, test in kf.split(df):
    
        #  Split data and labels for testing and training
        X_train = df.iloc[np.array(train)].drop(columns=['Prediction', "Email No."])
        y_train = df.iloc[np.array(train)]['Prediction'].values
        X_test = df.iloc[np.array(test)].drop(columns=['Prediction', "Email No."])
        y_test = df.iloc[np.array(test)]['Prediction'].values

        # Create model for kNN and retrieve model accuracy
        knn_model = KNeighborsClassifier(n_neighbors=k)
        knn_model.fit(X_train, y_train)
        y_pred = knn_model.predict(X_test)
        accuracy.append(knn_model.score(X_test, y_test))

    # Get average accuacy of each model
    accuracy = np.array(accuracy)
    avg_acc.append(np.mean(accuracy))

# Plot Avg Accuracy vs k
plt.plot(kVals, avg_acc, "-o")
plt.title("kNN 5-Fold Cross Validation")
plt.xlabel("k")
plt.ylabel("Average Accuracy")
plt.grid()
plt.show()


    
    