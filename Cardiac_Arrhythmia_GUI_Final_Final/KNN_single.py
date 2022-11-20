import numpy as np
from sklearn.neighbors import KNeighborsClassifier

def knn(data):
    text_file = open("data_2.txt", "r")
    x=0
    L=[]
    X=[]
    Y=[]
    D = {}
    f=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    while(x<420):
        L.append(list(text_file.readline().split("\t")))
        x=x+1
    text_file.close()
    for i in range(len(L)):
        L[i]=[float(x) for x in L[i]]
    #print(L)
    i=0

    l =[1,2,3]
    term =[]
    while(i<int(420)):
        X.append(L[i][0:163])
        Y.append(L[i][163])
        i=i+1
    neigh = KNeighborsClassifier(n_neighbors=3)
    print("Training dataset : X[0:5]", X[0:5])
    print("training dataset dimension: ", np.array(X).shape)
    print("training dataset's one sample dimension: " , np.array(X[0]).shape)
    neigh.fit(X, Y) 
    #pred = clf.predict([56,1,165,64,81,174,401,149,39,25,37,-17,31,0,53,0,48,0,0,0,24,0,0,0,0,0,0,0,64,0,0,0,24,0,0,0,0,0,0,32,24,0,0,0,40,0,0,0,0,0,0,48,0,0,0,0,0,0,0,0,0,0,0,0,44,20,0,0,24,0,0,0,0,0,0,0,60,0,0,0,20,0,0,0,0,0,0,0,24,52,0,0,16,0,0,0,0,0,0,0,32,52,0,0,20,0,0,0,0,0,0,0,44,48,0,0,32,0,0,0,0,0,0,0,48,44,0,0,32,0,0,0,0,0,0,0,48,40,0,0,28,0,0,0,0,0,0,0,48,0,0,0,28,0,0,0,0,0,0,-0.6,0.0,7.2,0.0,0.0,0.0,0.4,1.5,17.2,26.5,0.0,0.0,5.5,0.0,0.0,0.0,0.1,1.7,17.6,29.5,0.3,-1.6,0.9,0.0,0.0,0.0,-0.3,0.4,-1.5,1.3,0.1,-6.4,0.0,0.0,0.0,0.0,-0.3,-1.6,-15.3,-25.5,-0.3,0.0,4.2,-0.9,0.0,0.0,0.4,0.7,8.3,12.3,0.2,0.0,2.2,0.0,0.0,0.0,-0.2,0.8,6.6,11.7,0.4,0.0,1.0,-8.8,0.0,0.0,0.5,-0.6,-21.6,-26.8,0.4,0.0,2.6,-7.9,0.0,0.0,0.8,2.0,-16.4,1.2,0.0,0.0,5.8,-7.7,0.0,0.0,0.9,3.8,-5.7,27.7,-0.2,0.0,9.5,-5.0,0.0,0.0,0.5,2.6,11.8,34.6,-0.4,0.0,11.0,-2.4,0.0,0.0,0.4,2.6,21.6,43.4,-0.5,0.0,8.5,0.0,0.0,0.0,0.2,2.1,20.4,38.8])
    test= []
    i=0
    pred =[]
	# converting data from 1D to 2D array as predict function expects to receive data in 2D format
    querying_data = [data]
    print("Testing data : querying_data", querying_data)
    print("query data dimension : ", np.array(querying_data).shape)
    pred = neigh.predict(querying_data)
    print(pred[0])
    return pred[0]




