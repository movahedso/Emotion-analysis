import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier as KNN
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, f1_score
from autofeat import FeatureSelector
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression



random_state=100
n_components=500
max_iter=1000



def _import():
    data=pd.read_csv(filepath_or_buffer=r'/content/drive/MyDrive/emotions.csv')
    target=np.array(data['label'])
    features=data.drop('label',axis=1)
    return features,target



def scale(x_train,x_test,y_train):
    scaler=StandardScaler()
    x_train=scaler.fit_transform(x_train,y_train)
    x_test=scaler.transform(x_test)
    return x_train,x_test



def autofeat(data,target,random_state):
    x_train, x_test, y_train, y_test = train_test_split(data, target,
                                                        random_state=random_state,
                                                        shuffle=True, train_size=0.8)
    x_train, x_test = scale(x_train, x_test, y_train)
    rdt = FeatureSelector(problem_type='classification',featsel_runs=5,n_jobs=-1,verbose=4)
    x_train = rdt.fit_transform(x_train, y_train)
    x_test = rdt.transform(x_test)
    return x_train, x_test, y_train, y_test



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
print("Autofeat")
x_train,x_test,y_train,y_test=autofeat(data,target,random_state)
print("\nMLP:")
mlp(x_train,x_test,y_train,y_test,random_state)
print("\nSVM:")
svm(x_train,x_test,y_train,y_test,random_state)
print("\nNaive bayes:")
naiveB(x_train,x_test,y_train,y_test)
print("\nLogistic Regression")
logistic_reg(x_train,x_test,y_train,y_test,random_state,solver='newton-cg')
print("\nKNN")
knn(x_train,x_test,y_train,y_test)