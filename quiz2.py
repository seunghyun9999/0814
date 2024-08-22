import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from pandas.plotting import scatter_matrix
#
header=['sepal-length','sepal-width','petal-length','petal-width','class']
data = pd.read_csv('./data/2.iris.csv',names=header)
des=data.describe()
print(des)

plt.hist(data['sepal-length'])
plt.hist(data['sepal-width'])
plt.hist(data['petal-length'])
plt.hist(data['petal-width'])
plt.show()

array = data.values
X = array[:,0:4]
Y = array[:,4]
# 시각화랑 데이터 요약
scaler = MinMaxScaler(feature_range=(0,1))
rescaled_X = scaler.fit_transform(X)
(X_train,X_text,
 Y_train,Y_test)=train_test_split(rescaled_X,Y,test_size=0.2)

model =DecisionTreeClassifier()
model.fit(X_train,Y_train)
fold = KFold(n_splits=10,shuffle=True)
# 10번 시행해라 바꿔가면서
acc = cross_val_score(model,rescaled_X,Y,cv=fold,scoring='accuracy')
print(acc)

y_pred =model.predict(X_text)
print(confusion_matrix(y_pred,Y_test))
print(classification_report(y_pred,Y_test))
