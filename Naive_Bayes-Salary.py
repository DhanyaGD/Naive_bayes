""""1) Prepare a classification model using Naive Bayes
for salary data

Data Description:

age -- age of a person
workclass	-- A work class is a grouping of work
education	-- Education of an individuals
maritalstatus -- Marital status of an individulas
occupation	 -- occupation of an individuals
relationship --
race --  Race of an Individual
sex --  Gender of an Individual
capitalgain --  profit received from the sale of an investment
capitalloss	-- A decrease in the value of a capital asset
hoursperweek -- number of hours work per week
native -- Native of an individual
Salary -- salary of an individual
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

salary_train = pd.read_csv('C:/ExcelrData/Data-Science_Assignments/Naive_Bayes/SalaryData_Train.csv')
salary_test = pd.read_csv('C:/ExcelrData/Data-Science_Assignments/Naive_Bayes/SalaryData_Test.csv')
salary_train.columns
salary_test.columns
string_columns = ['workclass', 'education', 'maritalstatus', 'occupation', 'relationship', 'race', 'sex', 'native']

from sklearn import preprocessing

label_encoder = preprocessing.LabelEncoder()
for i in string_columns:
    salary_train[i] = label_encoder.fit_transform(salary_train[i])
    salary_test[i] = label_encoder.fit_transform(salary_test[i])

col_names = list(salary_train.columns)
train_X = salary_train[col_names[0:13]]
train_Y = salary_train[col_names[13]]
test_x = salary_test[col_names[0:13]]
test_y = salary_test[col_names[13]]

######### Naive Bayes ##############

# Gaussian Naive Bayes

from sklearn.naive_bayes import GaussianNB

Gmodel = GaussianNB()
train_pred_gau = Gmodel.fit(train_X, train_Y).predict(train_X)
test_pred_gau = Gmodel.fit(train_X, train_Y).predict(test_x)

train_acc_gau = np.mean(train_pred_gau == train_Y)
test_acc_gau = np.mean(test_pred_gau == test_y)
print("train_acc_gau : ", train_acc_gau)  # 0.795
print("test_acc_gau : ", test_acc_gau)  # 0.794

# Multinomial Naive Bayes

from sklearn.naive_bayes import MultinomialNB

Mmodel = MultinomialNB()
train_pred_multi = Mmodel.fit(train_X, train_Y).predict(train_X)
test_pred_multi = Mmodel.fit(train_X, train_Y).predict(test_x)

train_acc_multi = np.mean(train_pred_multi == train_Y)
test_acc_multi = np.mean(test_pred_multi == test_y)
print("train_acc_multi :", train_acc_multi)  # 0.772
print("test_acc_multi :", test_acc_multi)  # 0.774
