# -*- coding: utf-8 -*-
"""
Created on Thu Sep  8 09:03:09 2016

@author: sanjeev
"""
import pandas as pd
from process_data import encode_data
train=pd.read_csv('./train.csv')
test=pd.read_csv('./test.csv')

train_data, labels, pred_data, pred_ids, classes = encode_data(train, test)

#print train_data.head(1)
#print labels

#split the training data into training and test data
#here, the test data will be used to test the trained clasifier 
#to check the accuracy

from sklearn.cross_validation import StratifiedShuffleSplit
#from sklearn.cross_validation import train_test_split

## for stratified division of train data
splitted_data = StratifiedShuffleSplit(labels, 10, test_size=0.2, random_state=23)

for train_index, test_index in splitted_data:
    X_train, X_test = train_data.values[train_index], train_data.values[test_index]
    y_train, y_test = labels[train_index], labels[test_index]

##for normal division of train data
#normal division doesn't gives very encouraging result
#stratified diviision gives better result
#X_train,X_test,y_train,y_test=train_test_split(train_data,labels,test_size=0.3, random_state=42)

print len(X_test),len(X_train),len(train_data)

from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

clf1=SVC(kernel="rbf", C=0.025, probability=True)
#clf2=DecisionTreeClassifier()
clf2=LinearDiscriminantAnalysis()

clf1.fit(X_train,y_train)
clf2.fit(X_train,y_train)

prediction1=clf1.predict(X_test)
prediction2=clf2.predict(X_test)
pred_prob2=clf2.predict_proba(X_test)
from sklearn.metrics import accuracy_score,log_loss

accu1=accuracy_score(y_test,prediction1)
accu2=accuracy_score(y_test,prediction2)
ll = log_loss(y_test, pred_prob2)

print "SVM accracy :"+str(accu1)
print "Decision Tree Classifier :"+str(accu2)
print "Log Loss for econd classifier :"+str(ll)
pred_test_data=clf2.predict_proba(pred_data)

print pred_test_data


submission = pd.DataFrame(pred_test_data, columns=classes)
submission.insert(0, 'id', pred_ids)
#submission.reset_index()

submission.to_csv("submission.csv",index=False)
