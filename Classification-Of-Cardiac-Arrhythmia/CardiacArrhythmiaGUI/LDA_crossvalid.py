'''TO IMPLEMENT LDA'''
import warnings
warnings.filterwarnings("ignore")
import random
import copy
import sklearn.discriminant_analysis
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from tkinter import *

dataset=[]
X=[]
Y=[]
trainSet=[]
feat=279
k = 10

#TO GET DATASET
def trainAndTest():
    x=0
    global dataset
    text_file=open("ReplacedWithMedian.csv","r")
    while(x<452):
        dataset.append(list(text_file.readline().split(",")))
        x+=1
    text_file.close()
    for i in range(len(dataset)):
        dataset[i]=[float(x) for x in dataset[i]]
    random.shuffle(dataset)

#Cross Validation
def crossval():
    child = Tk()
    child.title("LDA Accuracy")
    child.geometry("300x300")

    trainAndTest()
    pos =0
    j=0
    k=10
    accuracy=0

    pos_x=60
    pos_y=50
    #trainer=[]
    global X
    global Y
    global dataset
    total = 452
    while(pos<=total):
        warnings.filterwarnings("ignore")
        tester=[]
        trainer=[]
        #print pos
        if((pos+int(total/k)) > total ):
            if(total-pos)<10:
                break
            tester = dataset[pos:total]
            trainer = list(set(tuple(x) for x in dataset) - set(tuple(x) for x in tester))
            n=k+1
            pos = total+1
        else:
            tester = dataset[pos:(((j+1)*(int(total/k))))]
            trainer = list(set(list(tuple(x) for x in dataset))-set(list(tuple(x) for x in tester)))
            trainer = list(list(x) for x in trainer)
            pos=int(((j+1)*(total/k)))
            n=k
        j+=1
        accuracy1 = lda(trainer,tester)
        accuracy1 = round(accuracy1,2)
        #la = Label(child, text="Hello")
        #la.pack()
        label1 = Label(child, text="Fold "+str(j)+": ", fg="red")
        label1.place(x=pos_x,y=pos_y)
        label2 = Label(child, text=str(accuracy1)+" %")
        label2.place(x=pos_x+60, y=pos_y)
        pos_y = pos_y+20

        print('Accuracy of fold ',j, ' is = ' ,accuracy1)
        accuracy+=accuracy1

        
    accuracy = float(accuracy)/float(n)
    print('Average accuracy of all ', n ,' folds = ', accuracy)
    accuracy = round(accuracy,2)
    label3 = Label(child, text="Average: ",fg="red")
    label3.place(x=pos_x, y=pos_y)
    label4 = Label(child, text=str(accuracy)+" %")
    label4.place(x=pos_x+60, y=pos_y)
    mainloop()

  
    

#Running LDA        
def lda(train,test):
    global feat
    labels =[]
    #print train[0]
    classes=[]
    train1=copy.deepcopy(train)
    test1=copy.deepcopy(test)
    copy1=copy.deepcopy(train)
    copy2=copy.deepcopy(test)
    for i in range(len(train)):
        del train1[i][feat]
    for i in range(len(test)):
        del test1[i][feat]
    i=0
    while(i<len(copy1)):
        classes.append(copy1[i][feat])
        i+=1;
    i=0
    while(i<len(copy2)):
        labels.append(copy2[i][feat])
        i+=1;
    #print train
    #Try 'auto' instead of None as second parameter to LDA for better accuracy
    neigh=LDA('lsqr',None)
    neigh.fit(train1, classes)
    pred=neigh.predict(test1)
    correct = 0
    '''print("Actual Value | Predicted Value")
    print("------------------------------")'''
    for i in range(len(labels)):
        if labels[i]==pred[i]:
                correct+=1
        #print ("\t{}\t{}\t".format(labels[i],pred[i]))
    #print((correct/float(len(copy1)))*100.0)
    return ((correct/float(len(labels)))*100.0)


def lda_main():
    crossval()



