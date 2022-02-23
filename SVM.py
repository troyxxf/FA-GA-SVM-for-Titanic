
#coding=utf-8


#Import necessary modules
import copy

print('Importing modules...')
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
import seaborn as sns
import Titanic_preprocess
import Titanic_featureSelection_GA as tfg
from sklearn.svm import SVC, LinearSVC


#Load the data and preprocess
print('Loading and preprocessing data...')
train_df = pd.read_csv('./train.csv')
test_df = pd.read_csv('./test.csv')
train_df, test_df = Titanic_preprocess.preprocess(train_df, test_df)






feature_combine=[1,1,1,1,1,1]
x_train = train_df.drop(['Survived'], axis=1)
x_train = x_train[x_train.columns[feature_combine]]
y_train = train_df.Survived
x_test = test_df[test_df.columns[feature_combine]]
svc = SVC()
svc.fit(x_train, y_train)
score = svc.score(x_train, y_train)
print("SVM分数",score)





# #Train the model by GA(feature selection) and SVM
#
# while True:
#     print('\n')
#     print('Generation: ', generation)
#     #Update train and test by each feature combination
#     feature_combines = tfg.feature_selection(population, num_features, feature_combines, fitness_ls)
#     fitness_ls = []
#     for feature_combine in feature_combines:
#         print(feature_combine)
#         x_train = train_df.drop(['Survived'], axis = 1)
#         x_train = x_train[x_train.columns[feature_combine]]
#         y_train = train_df.Survived
#         x_test = test_df[test_df.columns[feature_combine]]
#         #Support Vector Machines
#         svc = SVC()
#         try:
#             svc.fit(x_train, y_train)
#             #y_pred = svc.predict(x_test)
#             score = svc.score(x_train, y_train)
#             fitness_ls.append(score)
#             #print('Score: ', score)
#         except Exception as e:
#             print(e)
#             score = 0
#             fitness_ls.append(score)
#     print('Max score: ', max(fitness_ls))
#     print('Average score: ', sum(fitness_ls) / len(fitness_ls))
#     print('Min score: ', min(fitness_ls))
#     print(feature_combines[fitness_ls.index(max(fitness_ls))])
#     if max(fitness_ls) > 0.8 and sum(fitness_ls) / len(fitness_ls) > 0.77:
#         print(feature_combines[fitness_ls.index(max(fitness_ls))])
#         break
#     generation += 1

