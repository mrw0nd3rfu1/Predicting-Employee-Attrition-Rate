import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error,r2_score
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
import numpy as np

X = pd.read_csv('Train.csv')
X_test = pd.read_csv('Test.csv')
X_sample = pd.read_csv('Test.csv')

X.fillna(X.mean(),inplace=True)
Y= X.Attrition_rate
X.drop(['Attrition_rate','Hometown','Employee_ID'],axis=1,inplace=True)
###
X_test.drop(['Hometown','Employee_ID'],axis=1,inplace=True)

Col_to_encode = ['Gender','Relationship_Status','Unit','Decision_skill_possess','Compensation_and_Benefits']

X[Col_to_encode]= X[Col_to_encode].apply(lambda col:LabelEncoder().fit_transform(col))
###
X_test[Col_to_encode]= X_test[Col_to_encode].apply(lambda col:LabelEncoder().fit_transform(col))
X_test.fillna(X_test.mean(), inplace=True)
#train_X,val_X,train_Y,val_Y = train_test_split(X,Y,train_size=0.8,test_size=0.2,random_state=0)

#model = RandomForestRegressor(n_estimators=100, random_state=1)
model = LinearRegression()

model.fit(X,Y)
preds = model.predict(X_test)
output = pd.DataFrame({'Employee_ID':X_sample.Employee_ID,'Attrition_rate':preds})
output.to_csv('submission.csv', index=False)
# me = mean_absolute_error(val_Y,preds)
# r_score = r2_score(val_Y,preds)
# print(me,r_score)