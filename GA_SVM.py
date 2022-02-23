
#coding=utf-8


#Import necessary modules
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



#Initialize params for GA

generation = 0
#The number of population to start with for GA
population = 8
num_features = len(test_df.columns)
#i.e., feature_combines = [['True', 'False', 'False'], 
#['False', 'True', 'True']]
feature_combines = []
#fitness_ls is used to evaluate performance of each feature_combination
fitness_ls = []

MAXGEN=20
trace = np.zeros((MAXGEN, 2))

#Train the model by GA(feature selection) and SVM

while generation<MAXGEN:
    print('\n')
    print('Generation: ', generation)
    #Update train and test by each feature combination
    feature_combines = tfg.feature_selection(population, num_features, feature_combines, fitness_ls)
    fitness_ls = []
    for feature_combine in feature_combines:
        x_train = train_df.drop(['Survived'], axis = 1)
        x_train = x_train[x_train.columns[feature_combine]]
        y_train = train_df.Survived
        x_test = test_df[test_df.columns[feature_combine]]
        #Support Vector Machines
        svc = SVC()
        try:
            svc.fit(x_train, y_train)
            #y_pred = svc.predict(x_test)
            score = svc.score(x_train, y_train)
            fitness_ls.append(score)
            #print('Score: ', score)
        except Exception as e:
            print(e)
            score = 0
            fitness_ls.append(score)
    trace[generation, 0] = max(fitness_ls)
    trace[generation, 1] =(sum(fitness_ls)/len(fitness_ls))
    print('Max score: ', max(fitness_ls))
    print('Average score: ', sum(fitness_ls) / len(fitness_ls))
    print('Min score: ', min(fitness_ls))
    print(feature_combines[fitness_ls.index(max(fitness_ls))])

    # if max(fitness_ls) > 0.8 and sum(fitness_ls) / len(fitness_ls) > 0.77:
    #     print(feature_combines[fitness_ls.index(max(fitness_ls))])
    #     break
    generation += 1

x=np.arange(0,MAXGEN)
y1=trace[:,0]
y2=trace[:,1]
plt.plot(x, y1, 'r', label='optimal value')
plt.plot(x, y2, 'g', label='average value')
plt.xlabel("Iteration")
plt.ylabel("function value")
plt.title("Genetic Algorithm for function optimization")
plt.legend()
plt.show()