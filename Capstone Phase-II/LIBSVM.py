import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from libsvm.svmutil import *



data = pd.read_csv('/Users/sinchithahv/PycharmProjects/pythonProject37/CVD_combined (1).csv')
data = data.dropna()

x = data.drop('condition',axis=1)
y = data['condition']
print(x)

x = np.asarray(x).astype('float')
y = np.asarray(y)

x = (x.tolist())
y = (y.tolist())

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=43)

param = svm_parameter("-q")
param.kernel_type = 1

problem = svm_problem(y_train, x_train)

m = svm_train(problem,param)

x_test = list(x_test)
y_test = list(y_test)

pred_lbl, pred_acc, pred_val = svm_predict(y_test, x_test, m)
print(pred_lbl)
print(pred_acc)
print(pred_val)

results = []
for c in range(-5, 10):
    for g in range(4,0):
        for k in range(0,4):
            param.C, param.gamma = 2**c, 2**g
            m = svm_train(problem,param)
            p_lbl, p_acc, p_val = svm_predict(y_test,x_test,m)
            results.append([param.C, param.gamma, p_acc[0]])
bestIdx = np.argmax(np.array(results)[0])
n = bestIdx
print("best accuracy is at index",n)
print('best parameter',results[bestIdx])

Result = np.array(results)

param.C = Result[bestIdx,0]
param.gamma = Result[bestIdx,1]
ker=Result[bestIdx,2]

m = svm_train(problem, param)
pred_lbl, pred_acc, pred_val = svm_predict(y_test,x_test,m)

