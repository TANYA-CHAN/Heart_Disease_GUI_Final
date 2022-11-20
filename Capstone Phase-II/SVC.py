import pandas as pd
import sklearn.svm
from sklearn.svm import _libsvm
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import numpy as np




data = pd.read_csv('/Users/sinchithahv/PycharmProjects/pythonProject37/CVD_combined (1).csv')
data = data.dropna()

x = data.drop('condition',axis=1)
print(x)
y = data['condition']
print(y)
#x = np.asarray(x).astype('float')
#y = np.asarray(y)

#x = (x.tolist())
#y = (y.tolist())

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.33,random_state=43)

print(x_train)
print(y_train)

my_classes = np.unique(y)
clfs = []
for k in my_classes:
    my_model = SVC(gamma = 'auto', C=1, kernel='rbf',class_weight='balanced',probability=True, random_state=42)
    clfs.append(my_model.fit(x_train, y_train==k))

#Prediction
prob_table = np.zeros((len(x_test),len(my_classes)))

for i,clf in enumerate(clfs):
    probs = clf.predict_proba(x_test)[:,1]
    prob_table[:,i] = probs

y_pred = my_classes[prob_table.argmax(1)]
print("Test accuracy = ",accuracy_score(y_test,y_pred)*100)











