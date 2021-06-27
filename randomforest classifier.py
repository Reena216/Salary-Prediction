# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 17:07:57 2019

@author: amris
"""

import pandas as pn
import numpy as nm
dp =pn.read_csv(r'C:\Users\amris\Downloads\SalaryData_Train.csv')
dp.columns
y=pn.DataFrame(dp.Salary)
x=dp.drop(['educationno', 'maritalstatus','occupation', 'relationship', 'race', 'sex', 'capitalgain','capitalloss', 'hoursperweek', 'native', 'Salary'],axis=1)

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
age=le.fit_transform(dp.age)
workclass= le.fit_transform(dp.workclass)
education= le.fit_transform(dp.education)
salary=le.fit_transform(dp.Salary)
dp1=pn.DataFrame(age,columns=['age'])
dp1['workclass']=workclass
dp1['education']=education
dp2=pn.DataFrame(salary,columns=['salary'])

from sklearn.ensemble import RandomForestClassifier
obj= RandomForestClassifier()
obj.fit(dp1,dp2)
pr=obj.predict(dp1)
from sklearn.metrics import accuracy_score
accuracy_score(dp2,pr)*100
