'''VFI5: CONSIDERING AL LOWER BOUNDS AS POINT INTERVALS'''
import random
import math
from tkinter import *


    
#GLOBAL VARIABLES:
D = {}
e =  2.71828
DMV = {}
total = 420
train_no = int(0.9*total)
test_no = total -train_no
features_no = 163
class_total = 16
L=[]
prior=[]
P ={}
tester =[]
labels=[]
T=[]
Endpoints=[]
Classdist=[]
Ptdist=[]
k = 2
#TO CROSS VALIDATE
def crossval():
    child = Tk()
    child.title("VFI5 Accuracy")
    child.geometry("300x300")

    dataset()
    accuracy =0
    global train_no
    pos =0
    j=0
    pos_x=60
    pos_y=50
    while(pos<=total):
        #print pos
        if((pos+int(total/k)) > total ):
            if(total-pos)<10:
                break
            tester1 = L[pos:total]
            n=k+1
            pos = total+1
        else:
            tester1 = L[pos:(((j+1)*(int(total/k))))]
            pos=int(((j+1)*(total/k)))
            n=k
        j+=1
        trainSet = list(set(tuple(x) for x in L) - set(tuple(x) for x in tester1))
        train_no = len(trainSet)
        datatrain(trainSet)
        test_data(tester1)
        endpoints()
        countinterval()
        accuracy1 = test()
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

#TO GET TEST DATASET IN FORM OF LIST
def dataset():
    text_file = open("data_2.txt", "r")
    x=0
    X=[]
    Y=[]
    while(x<total):
        L.append(list(text_file.readline().split("\t")))
        x=x+1
    text_file.close()
    for i in range(len(L)):
        L[i]=[float(x) for x in L[i]]
    #print(L)
    i=0
    random.shuffle(L)

#TO GET TRAIN DATA IN DICTIONARY
def datatrain(L):
    i=0
    f=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    while(i<train_no):
        index = int(((L[i][features_no])-1))
        if(f[index]==0):
            P = [L[i][0:features_no]]
            D.update({index:P})
            f[index]=1
        else:
            D[index].append(L[i][0:features_no])
        i=i+1
    for i in range(0,class_total):
        if(f[i]==0):
            P = []
            D.update({i:P})
            f[i]=1
    #print(D)

#TO GET ENDPOINTS OF EACH CLASS OF EACH FEATURE

def endpoints():
    global Endpoints
    Endpoints=[]
    for k in range(features_no):
        Endpoints1=[]
        Endpoints1.append(-1000)
        for i in range(class_total):
            L=[]
            for j in range(len(D[i])):
                if(D[i][j][k]!=999):
                    L.append(D[i][j][k])
            #print(L)
            if(len(L)!=0):
                Endpoints1.append(min(L))
                Endpoints1.append(max(L))
        Endpoints1.append(max(Endpoints1)+1000)
        Endpoints.append(Endpoints1)
    #print(Endpoints)
    for i in range(features_no):
        Endpoints[i] = list(set(Endpoints[i]))
        Endpoints[i].sort()
    #print(Endpoints)
    #print(Endpoints[0])
    #print Ptintervals

#To count no of instances of each class in each interval of each feature
def countinterval():
    global Classdist, Ptdist
    Ptdist =[]
    Classdist =[]
    for k in range(features_no):
        #print(Endpoints[k])
        Classdist1 =[]
        #print(len(Endpoints[0]))
        Ptdist1={}
        for i in (Endpoints[k]):
            Ptdist1.update({str(i):[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0] })  
        
        for i in range(0, len(Endpoints[k])):
            Classdist1.append([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
        for i in range(class_total):
            for j in range (len(D[i])):
                for l in range((len(Endpoints[k])-1)):
                    #print(int(Endpoints[k][l]))
                    #print(int(Endpoints[k][l+1]))
                    #print(D[i][j][k])
                    if(D[i][j][k]!=999):
                        if(D[i][j][k] in Endpoints[k]):
                            Ptdist1[str(D[i][j][k])][i]+=1
                        elif( D[i][j][k] in range(int(Endpoints[k][l]) , int(Endpoints[k][l+1]))):
                            #print(Classdist[l][i])
                            Classdist1[l][i] = Classdist1[l][i] +1
        #print(Classdist1)
        Classdist.append(Classdist1)
        Ptdist.append(Ptdist1)
    #print(Ptdist)
    #print Classdist
    #print(Endpoints[1])
    #print(Classdist[1][0][0])
    for k in range(features_no):
        for i in range(0,class_total):
            for j in range(0,len(Endpoints[k])):
                if(len(D[i])!=0):
                    #print(k , i ,j)
                    Classdist[k][j][i] = float(Classdist[k][j][i])/float( len(D[i]))
    for j in range(features_no):
        for i in range(class_total):
            for k in (Ptdist[j].keys()):
                if(len(D[i])!=0):
                    Ptdist[j][k][i]=  float(Ptdist[j][k][i])/float( len(D[i]))
                
    s=[]
    for k in range(features_no):
        s1=[]
        for i in range(0,len(Endpoints[k])):
            s1.append(sum(Classdist[k][i]))
        s.append(s1)

    for k in range(features_no):
        for i in range(0,class_total):
            for j in range(0,len(Endpoints[k])):
                if(s[k][j]!=0):
                    #print(k , i ,j)
                    Classdist[k][j][i] = float(Classdist[k][j][i])/float(s[k][j])
    s2=[]
    for k in range(features_no):
        s1={}
        for i in (Ptdist[k].keys()):
            s1.update({i:sum(Ptdist[k][i])})
        s2.append(s1)
    for j in range(features_no):
        for k in (Ptdist[j].keys()):
            for i in range(class_total):
                if(sum(Ptdist[j][k])!=0):
                    Ptdist[j][k][i]=  float(Ptdist[j][k][i])/float(s2[j][k])



#TO FIND INTERVAL OF A TEST FEATURE
def find_interval(f,ef):
    if(ef in Endpoints[f]):
        m = -9999
        #print 'ok'
        for k in range(class_total):
            if m<Ptdist[f][str(ef)][k] :
                m = Ptdist[f][str(ef)][k]
                classno = k
        return [classno,str(ef),0]
                     
    else:        
        for i in range((len(Endpoints[f])-1)):
            if(ef<Endpoints[f][i+1] and ef>Endpoints[f][i]):
                #print Endpoints[f][i]
                #print Endpoints[f]
                return [i]
            '''elif (ef==Endpoints[f][i]):
                if(ef == Endpoints[f][1]):
                    return [i]
                elif (ef==Endpoints[f][-2]):
                    return [i-1]
                else:
                    return [i, i-1]'''




#To classify one test case
def classify(ex):
    #print("ok")
    Vote=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    #print(Vote)
    for i in range(class_total):
        for j in range(features_no):
            Vote1=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
            interval = find_interval(j, ex[j])
            if(ex[j]!=999 and interval!=None):
                #print tester[i][j]
                for k in range(class_total):
                    #print( j , interval , k)
                    if(len(D[k])!=0):
                        if(len(interval) ==1):
                            Vote1[k] += float(Classdist[j][int(interval[0])][k])
                        elif(len(interval) ==2):
                            Vote1[k] += (float(Classdist[j][int(interval[0])][k]) + float(Classdist[j][int(interval[1])][k]))/2.0
                        elif(len(interval) ==3):
                            Vote1[k] += float(Ptdist[j][str(ex[j])][k])
                s=sum(Vote1)
                for m in range(class_total):
                    if(sum(Vote1)!=0):
                        Vote1[m]=float(Vote1[m])/s
                for m in range(class_total):
                    Vote[m]+=Vote1[m]
                
    #print(Vote)
    m = max(Vote)
    #print m
    for i in range(class_total):
        if Vote[i] == m:
            pos =i
    return pos+1



#TO CLASSIFY ALL TEST CASES
def test():
    T=[]
    for i in range(len(tester)):
        T.append(classify(tester[i]))
    count =0
    for i in range(len(T)):
        if(T[i] == int(labels[i])):
            count= count+1
        #print(T[i],labels[i])
    #print count
    #print int(len(labels))
    accuracy = (float(count)/ float(len(labels)))*100
    #print("acc",acuracy)
    return accuracy              
                   

#TO GET TEST DATA SET
def test_data(L):
    global labels
    labels =[]
    global tester
    tester =[]
    for i in range(len(L)):
        tester.append(L[i][:features_no])
        labels.append(L[i][features_no])
    #print(tester)

def vfi():
    crossval()
