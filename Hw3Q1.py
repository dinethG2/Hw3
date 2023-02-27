import numpy as np  
import matplotlib.pyplot as plt


# Determine the label of the test data point
def testLabel(point):
    distance = []
    for i in range(len(data)):
        distance.append((np.sqrt((data[i][0]-point[0])**2 + (data[i][1]-point[1])**2), i))
   
    return(data[min(distance)[1]][2])
    
def hw3_1():    
    # Split array into its classes
    Array0 = data[data[:,2] == 0]
    Array1 = data[data[:,2] == 1]

    # Set axis limits
    plt.xlim(-2, 2)
    plt.ylim(-2, 2)

    # Create test data
    Test_data = []
    for i in np.arange(-2,2.1, 0.1):
        for j in np.arange(-2,2.1, 0.1):
            Test_data.append([i,j])

    Test_data = np.array(Test_data)

    # Plot test data
    for i in Test_data:
        if(testLabel(i) == 1):
            plt.scatter(i[0], i[1], color = "red", s = 10, marker = ".")
           
        else:
            plt.scatter(i[0], i[1], color = "blue", s = 10, marker = ".")
       


    # Plot training data
    plt.scatter(Array0[:,0], Array0[:,1], marker='o', color = "black", s=10)
    plt.scatter(Array1[:,0], Array1[:,1], marker='+', color = 'black', s=10)
    plt.show()


# Question 1
# Text file data converted to integer data type
data = np.loadtxt("data/D2z.txt", dtype=float)
hw3_1()