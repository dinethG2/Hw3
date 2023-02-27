from matplotlib import pyplot as plt
import pandas as pd  
import numpy as np  
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression

# Used to plot the ROC of each model
def plotROC(y_test, c):

    # Set up data for ROC plot with labels and confidence values
    instances = []
    for yi,ci in zip(y_test,c):
        instances.append([yi,ci])
    instances = np.array(instances)
    instances = instances[instances[:, 1].argsort()][::-1]

    # Initialize variables for plotting on ROC
    num_pos = instances[:,0].sum()
    num_neg = len(instances) - num_pos
    TP = 0
    FP = 0
    last_TP = 0
    TPRlist = []
    FPRlist = []
    FPRlist.append(0)
    TPRlist.append(0)

    for i in range(len(instances)):

        # Find high to low label splits and create data point for plotting
        if(i>1 and (instances[i][1] != instances[i-1][1]) and (instances[i][0]==0) and (TP>last_TP)):
            FPR = FP/num_neg
            TPR = TP/num_pos
            FPRlist.append(FPR)
            TPRlist.append(TPR)
            last_TP = TP

        # Increment counters
        if(instances[i][0] == 1):
            TP+=1
        else:
            FP+=1

    # Update variables
    FPR = FP/num_neg
    TPR = TP/num_pos
    FPRlist.append(FPR)
    TPRlist.append(TPR)

    #  Plot the ROC
    plt.plot(FPRlist, TPRlist)
    plt.grid()
    plt.xlabel("False Positive Label (Positive Label: 1)")
    plt.ylabel("True Positive Label (Positive Label: 1)")




# Question 5
# making dataframes
df = pd.read_csv("data/emails.csv") 
df_train = df.iloc[:4000,:]
df_test = df.iloc[4000:,:]


#  Create X,y train and testing datasets
X_train = df_train.drop(columns=['Prediction', "Email No."])
y_train = df_train['Prediction'].values

X_test = df_test.drop(columns=['Prediction', "Email No."])
y_test = df_test['Prediction'].values

handles = []
labels = []

# KNN
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, y_train)
y_pred = knn_model.predict(X_test)

# Confidence of label predicted as 1
c = knn_model.predict_proba(X_test)[:,1]
plotROC(y_test, c)

#calculate AUC of model
auc1 = metrics.roc_auc_score(y_test, c)



# Logistic Regression
log_model = LogisticRegression(random_state=0, max_iter = 1000)
log_model.fit(X_train, y_train)
y_pred = log_model.predict(X_test)  

# # Confidence of label predicted as 1
c = log_model.predict_proba(X_test)[:,1]
plotROC(y_test, c)


#calculate AUC of model
auc2 = metrics.roc_auc_score(y_test, c)


plt.legend(loc="lower right", labels = [f"KNN (AUC = {auc1:.2f})", f"logistic regression  (AUC = {auc2:.2f})"])
plt.grid()
plt.show()