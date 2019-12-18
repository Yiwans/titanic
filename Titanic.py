import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Imputer

# Algorithms
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score

#------------------------------ ----------------------
#-------------------Data_imshow ---------------
#------------------------------ ----------------------

data_train = pd.read_csv('train.csv')
data_test = pd.read_csv('test.csv')
data_all = pd.concat([data_train, data_test])
print('----------------------------------------------------------------------------')
print( data_all[["Pclass","Survived"]].groupby(["Pclass"], as_index = False).mean())
print('----------------------------------------------------------------------------')
print( data_all[["Sex","Survived"]].groupby(["Sex"], as_index = False).mean())
print('----------------------------------------------------------------------------')
data_all['FamilySize'] = data_all['SibSp'] + data_all['Parch'] + 1
print( data_all[["FamilySize","Survived"]].groupby(["FamilySize"], as_index = False).mean())
print('----------------------------------------------------------------------------')
print( data_all[["Embarked","Survived"]].groupby(["Embarked"], as_index = False).mean() )
print('----------------------------------------------------------------------')

#------------------------------ ----------------------
#-------------------Data_preprocessing ---------------
#------------------------------ ----------------------
#-----------------------Missing ----------------------
N_begin = data_all.isnull().sum()
total_cells = np.product(data_test.shape) + np.product(data_train.shape)
total_missing = data_train.isnull().sum() + data_test.isnull().sum()
Per = (total_missing/total_cells) * 100
print(Per)
print('----------------------------------------------------------------------')
#--------------------- Embarked ----------------------
data_all.Embarked.fillna('S', inplace = True)
#----------------------Cabin--------------------------
data_all.Cabin.fillna('NaN', inplace = True)
#----------------------Fare--------------------------
imputer = Imputer(missing_values='NaN',strategy='median',axis=1)
new = imputer.fit_transform(data_all.Fare.values.reshape(1,-1))
data_all['Fare'] = new.T
N_final= data_all.isnull().sum()
data_all['Fare_category'] = pd.qcut(data_all['Fare'], 4)
print( data_all[["Fare_category","Survived"]].groupby(["Fare_category"], as_index = False).mean())
print('----------------------------------------------------------------------')
#--------------------- Title ----- --------------------
data_all['Title'] = data_all.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
data_all['Title'] = data_all['Title'].replace(['Mlle',' Ms'], 'Miss')
data_all['Title'] = data_all['Title'].replace(['Mme','Lady', 'Dona','Countess'], 'Mrs')
data_all['Title'] = data_all['Title'].replace(['Jonkheer','Sir','Don','Dr','Capt','Col','Major','Rev'], 'Mr')
#------------------------Age -------------------------
#imputer = Imputer(missing_values='NaN',strategy='median',axis=1)
#new = imputer.fit_transform(data_all.Age.values.reshape(1,-1))
#data_all['Age'] = new.T
mean = data_all["Age"].mean()
std = data_all["Age"].std()
is_null = data_all["Age"].isnull().sum()
rand_age = np.random.randint(mean - std, mean + std, size = is_null)
age_slice = data_all["Age"].copy()
age_slice[np.isnan(age_slice)] = rand_age
data_all["Age"] = age_slice
data_all["Age"] = data_all["Age"].astype(int)
data_all["Age"].isnull().sum()

data_all['Age_category'] = pd.cut(data_all['Age'], 4)
print( data_all[["Age_category","Survived"]].groupby(["Age_category"], as_index = False).mean() )
print('----------------------------------------------------------------------')

#------------------------------ ----------------------
#--------------Categ signs encoding-------------
#------------------------------ ----------------------
data_all['Sex'] = LabelEncoder().fit_transform(data_all['Sex'])
data_all['Title'] = LabelEncoder().fit_transform(data_all['Title'])
data_all['Age_category'] = LabelEncoder().fit_transform(data_all['Age_category'])
data_all['Embarked'] = LabelEncoder().fit_transform(data_all['Embarked'])
data_all['Fare_category'] = LabelEncoder().fit_transform(data_all['Fare_category']);
#pd.get_dummies(data_all.Embarked, prefix="Emb", drop_first = True);
data_all.drop(['Name', 'Ticket', 'Cabin', 'SibSp', 'Parch', 'Fare','Age'], axis=1, inplace=True);
##------------------------------ ----------------------
##---------------------Data_preparing---------------
##------------------------------ ----------------------
train_dataset = data_all[:len(data_train)]
test_dataset = data_all[len(data_train):]
test_dataset.drop(['Survived'], axis=1, inplace=True)
Y_train = train_dataset["Survived"]
X_train  = train_dataset.drop(['PassengerId','Survived'], axis=1).copy()
X_test  = test_dataset.drop('PassengerId', axis=1)
##------------------------------ ----------------------
##---------------------RandForestClassif---------------
##------------------------------ ----------------------
RFClassif = RandomForestClassifier(criterion='entropy',n_estimators=600,
                            min_samples_split = 20,
                            min_samples_leaf=1,
                            max_features='auto',
                            oob_score=True,
                            random_state=1,
                            n_jobs=-1)
x_train, x_test, y_train, y_test = train_test_split(X_train, Y_train, test_size=0.3)

RFClassif.fit(x_train, np.ravel(y_train))
print("RF_accur: " + repr(round(RFClassif.score(x_test, y_test) * 100, 2)) + "%")
result_rf = cross_val_score(RFClassif,x_train,y_train,cv=10,scoring='accuracy')
print('CrossValidScore',round(result_rf.mean()*100,2))
y_pred = cross_val_predict(RFClassif,x_train,y_train,cv=10)
##------------------------------ ----------------------
##---------------------Result_predict---------------
##------------------------------ ----------------------
result = RFClassif.predict(X_test)
submission = pd.DataFrame({'PassengerId':test_dataset.PassengerId,'Survived':result})
submission.Survived = submission.Survived.astype(int)
print(submission.shape)
filename = 'Titanic_final.csv'
submission.to_csv(filename,index=False)
print('Saved file: ' + filename)

