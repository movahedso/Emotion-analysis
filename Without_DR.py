import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier as KNN
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, f1_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB



random_state=42
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



def preprocess(data,target,random_state):
    x_train,x_test,y_train,y_test=train_test_split(data,target,random_state=random_state,
                                                   shuffle=True,train_size=0.8)
    #x_train, x_test = scale(x_train, x_test, y_train)
    return x_train,x_test,y_train,y_test



def mlp(x_train,x_test,y_train,y_test,random_state):
    clf=MLPClassifier(random_state=random_state,max_iter=1000)
    clf.fit(x_train,y_train)
    y_pred=clf.predict(x_test)
    print("Accuracy: ",accuracy_score(y_test,y_pred))
    print("Confusion matrix:\n",confusion_matrix(y_test,y_pred))
    print("Classification report: \n",classification_report(y_test,y_pred))
    print("AUC: \n", roc_auc_score(y_test, clf.predict_proba(x_test), multi_class='ovo')*100)
    print("f1_score: \n", f1_score(y_test, y_pred, average='macro')*100)



def naiveB(x_train,x_test,y_train,y_test):
    clf=GaussianNB()
    clf.fit(x_train,y_train)
    y_pred=clf.predict(x_test)
    print("Accuracy: ", accuracy_score(y_test, y_pred))
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification report: \n", classification_report(y_test, y_pred))
    print("AUC: \n", roc_auc_score(y_test, clf.predict_proba(x_test), multi_class='ovo')*100)
    print("f1_score: \n", f1_score(y_test, y_pred, average='macro')*100)



def knn(x_train,x_test,y_train,y_test):
    clf=KNN()
    clf.fit(x_train,y_train)
    y_pred=clf.predict(x_test)
    print("Accuracy: ", accuracy_score(y_test, y_pred))
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification report: \n", classification_report(y_test, y_pred))
    print("AUC: \n", roc_auc_score(y_test, clf.predict_proba(x_test), multi_class='ovo')*100)
    print("f1_score: \n", f1_score(y_test, y_pred, average='macro')*100)



def svm(x_train,x_test,y_train,y_test,random_state):
    clf=SVC(random_state=random_state,kernel='rbf',probability=True)
    clf.fit(x_train,y_train)
    y_pred=clf.predict(x_test)
    print("Accuracy: ", accuracy_score(y_test, y_pred))
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification report: \n", classification_report(y_test, y_pred))
    print("AUC: \n", roc_auc_score(y_test, clf.predict_proba(x_test), multi_class='ovo')*100)
    print("f1_score: \n", f1_score(y_test, y_pred, average='macro')*100)



def logistic_reg(x_train,x_test,y_train,y_test,random_state,solver='lbfgs'):
    clf=LogisticRegression(random_state=random_state,max_iter=1000,solver=solver)
    clf.fit(x_train,y_train)
    y_pred=clf.predict(x_test)
    print("Accuracy: ", accuracy_score(y_test, y_pred))
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification report: \n", classification_report(y_test, y_pred))
    print("AUC: \n", roc_auc_score(y_test, clf.predict_proba(x_test), multi_class='ovo')*100)
    print("f1_score: \n", f1_score(y_test, y_pred, average='macro')*100)



data,target=_import()


print("-----------------------------------------------------------------")
print("Without DR")
x_train,x_test,y_train,y_test=preprocess(data,target,random_state)
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