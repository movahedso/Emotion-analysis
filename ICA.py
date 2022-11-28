import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier as KNN
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score, roc_auc_score
from sklearn.decomposition import FastICA
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression



random_state=100
n_components=50
max_iter=1000



def _import():
    data=pd.read_csv(filepath_or_buffer=r'C:\Users\Sobhan Movahedi\Desktop\emotions project\emotions.csv')
    target=np.array(data['label'])
    features=data.drop('label',axis=1)
    return features,target



def scale(x_train,x_test,y_train):
    scaler=StandardScaler()
    x_train=scaler.fit_transform(x_train,y_train)
    x_test=scaler.transform(x_test)
    return x_train,x_test



def ica(data,target,n_components,random_state,max_iter):
    x_train,x_test,y_train,y_test=train_test_split(data,target,random_state=random_state,
                                                   shuffle=True,train_size=0.8)
    x_train, x_test = scale(x_train, x_test, y_train)
    rdt=FastICA(n_components=n_components,random_state=random_state,max_iter=max_iter,whiten='unit-variance')
    x_train=rdt.fit_transform(x_train,y_train)
    x_test=rdt.transform(x_test)
    return x_train,x_test,y_train,y_test


def mlp(x_train,x_test,y_train,y_test,random_state):
    clf=MLPClassifier(random_state=random_state,max_iter=1000)
    clf.fit(x_train,y_train)
    y_pred=clf.predict(x_test)
    print("Accuracy: ",accuracy_score(y_test,y_pred))
    print("Confusion matrix:\n",confusion_matrix(y_test,y_pred))
    print("Classification report: \n",classification_report(y_test,y_pred))
    print("AUC: \n", roc_auc_score(y_test, clf.predict_proba(x_test), multi_class='ovo'))
    print("f1_score: \n", f1_score(y_test, y_pred, average='macro'))



def naiveB(x_train,x_test,y_train,y_test):
    clf=GaussianNB()
    clf.fit(x_train,y_train)
    y_pred=clf.predict(x_test)
    print("Accuracy: ", accuracy_score(y_test, y_pred))
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification report: \n", classification_report(y_test, y_pred))
    print("AUC: \n", roc_auc_score(y_test, clf.predict_proba(x_test), multi_class='ovo'))
    print("f1_score: \n", f1_score(y_test, y_pred, average='macro'))



def knn(x_train,x_test,y_train,y_test):
    clf=KNN()
    clf.fit(x_train,y_train)
    y_pred=clf.predict(x_test)
    print("Accuracy: ", accuracy_score(y_test, y_pred))
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification report: \n", classification_report(y_test, y_pred))
    print("AUC: \n", roc_auc_score(y_test, clf.predict_proba(x_test), multi_class='ovo'))
    print("f1_score: \n", f1_score(y_test, y_pred, average='macro'))



def svm(x_train,x_test,y_train,y_test,random_state):
    clf=SVC(random_state=random_state,kernel='rbf',probability=True)
    clf.fit(x_train,y_train)
    y_pred=clf.predict(x_test)
    print("Accuracy: ", accuracy_score(y_test, y_pred))
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification report: \n", classification_report(y_test, y_pred))
    print("AUC: \n", roc_auc_score(y_test, clf.predict_proba(x_test), multi_class='ovo'))
    print("f1_score: \n", f1_score(y_test, y_pred, average='macro'))



def logistic_reg(x_train,x_test,y_train,y_test,random_state,solver='lbfgs'):
    clf=LogisticRegression(random_state=random_state,max_iter=1000,solver=solver)
    clf.fit(x_train,y_train)
    y_pred=clf.predict(x_test)
    print("Accuracy: ", accuracy_score(y_test, y_pred))
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification report: \n", classification_report(y_test, y_pred))
    print("AUC: \n", roc_auc_score(y_test, clf.predict_proba(x_test), multi_class='ovo'))
    print("f1_score: \n", f1_score(y_test, y_pred, average='macro'))


data,target=_import()


print("-----------------------------------------------------------------")
print("Indepedent Component Analysis")
x_train,x_test,y_train,y_test=ica(data,target,n_components,random_state,max_iter)
print("\nMLP:")
mlp(x_train,x_test,y_train,y_test,random_state)
print("\nNaive Bayes:")
naiveB(x_train,x_test,y_train,y_test)
print("\nSVM:")
svm(x_train,x_test,y_train,y_test,random_state)
print("\nLogistic Regression")
logistic_reg(x_train,x_test,y_train,y_test,random_state)
print("\nKNN")
knn(x_train,x_test,y_train,y_test)