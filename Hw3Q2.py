import pandas as pd  
import numpy as np  
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold

# Question 2
# making dataframe 
df = pd.read_csv("data/emails.csv") 
   
# Split into 5 folds for cross validation
kf = KFold(n_splits=5)
i = 0
for train, test in kf.split(df):
   
    #  Split data and labels for testing and training
    X_train = df.iloc[np.array(train)].drop(columns=['Prediction', "Email No."])
    y_train = df.iloc[np.array(train)]['Prediction'].values
    X_test = df.iloc[np.array(test)].drop(columns=['Prediction', "Email No."])
    y_test = df.iloc[np.array(test)]['Prediction'].values

    
    knn_model = KNeighborsClassifier(n_neighbors=1)
    knn_model.fit(X_train, y_train)
    y_pred = knn_model.predict(X_test)
    
    # Calculate precison and recall
    precision = metrics.precision_score(y_test, y_pred)
    recall = metrics.recall_score(y_test, y_pred)

    i+=1
    print("Fold: ", i,", Test:", test[0], "...", test[len(test)-1])
    print("Accuracy: ", knn_model.score(X_test, y_test))
    print("Precision: ", precision)
    print("Recall: ", recall)
    print()
    