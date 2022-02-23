
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

#Initialize params
generation = 0
#The number of population to start with for GA
sizepop = 8
num_features = len(test_df.columns)
#i.e., feature_combines = [['True', 'False', 'False'],
#['False', 'True', 'True']]
feature_combines = []
#fitness_ls is used to evaluate performance of each feature_combination
fitness_ls = []



class FAIndividual:
    def __init__(self,vardim,gene):
        self.vardim=vardim
        self.gene=gene
        self.fitness=0
        self.chrom=None
    def generate(self):
        chooseNum=0
        while chooseNum<1:
            self.chrom = []
            for i in range(len(self.gene)):
                self.chrom.append( random.choice((True,False))  )
            sum=0
            for i in self.chrom:
                if i==True:
                    sum=1
            chooseNum=sum
    def calculateFitness(self):
        feature_combine=self.chrom
        x_train = train_df.drop(['Survived'], axis=1)
        x_train = x_train[x_train.columns[feature_combine]]
        y_train = train_df.Survived
        x_test = test_df[test_df.columns[feature_combine]]
        svc = SVC()
        try:
            svc.fit(x_train, y_train)
            score = svc.score(x_train, y_train)
            self.fitness=score
        except Exception as e:
            print(e)
            score = 0
            fitness_ls.append(score)

#萤火虫算法：
class FireflyAlgorithm:
    def __init__(self,sizepop,vardim,gene,MAXGEN,params):
        '''
        :param sizepop:种群数量
        :param vardim: 维度
        :param gene: 基因--任务调度顺序
        :param MAXGEN: 最大循环次数
        :param pareams: 参数 [beta,gamma,alpha]
        '''
        self.sizepop=sizepop
        self.vardim = vardim
        self.gene = gene
        self.MAXGEN = MAXGEN
        self.params = params
        self.population = [FAIndividual for i in range(self.sizepop)]
        self.fitness = np.zeros((self.sizepop, 1))
        self.trace = np.zeros((self.MAXGEN, 2))
    def initialize(self):
        for i in range(0,self.sizepop):
            ind=FAIndividual(self.vardim,self.gene)
            ind.generate()
            ind.calculateFitness()
            self.population[i]=ind
            self.fitness[i]=ind.fitness
    def evaluate(self):
        #evaluation of the population fitnesses
        for i in range(0, self.sizepop):
            self.population[i].calculateFitness()
            self.fitness[i]=self.population[i].fitness
    def move(self):
        for i in range(0,self.sizepop):
            for j in range(0,self.sizepop):
                if self.population[i].fitness<self.population[j].fitness:
                    for n in range(self.vardim):
                        if(self.population[i].chrom[n]!=self.population[j].chrom[n]):
                            touzi=random.random()
                            if(touzi<=self.params[1]):
                                self.population[i].chrom[n]=self.population[j].chrom[n]
                    self.population[i].calculateFitness()
                    self.fitness[i]=self.population[i].fitness
    # def randomWind(self):
        for i in range(0,self.sizepop):
            touzi=random.random()
            if(touzi<self.params[2]):
                chooseNum = 0
                while chooseNum < 1:
                    chrom = []
                    for i in range(len(self.gene)):
                        chrom.append(random.choice((True, False)))
                    sum = 0
                    for i in chrom:
                        if i == True:
                            sum = 1
                    self.population[i].chrom = chrom
                    chooseNum = sum

    def printResult(self):
        x=np.arange(0,self.MAXGEN)
        y1=self.trace[:,0]
        y2=self.trace[:,1]
        plt.plot(x, y1, 'r', label='optimal value')
        plt.plot(x, y2, 'g', label='average value')
        plt.xlabel("Iteration")
        plt.ylabel("function value")
        plt.title("Firefly Algorithm for function optimization")
        plt.legend()
        plt.show()
    def solve(self):
        '''
        evolution process of firefly algorithm
        '''
        self.t = 0
        self.initialize()
        self.evaluate()
        best = np.max(self.fitness)
        bestIndex = np.argmax(self.fitness)
        self.best=self.population[bestIndex]
        self.avefitness = np.mean(self.fitness)
        self.trace[self.t,0]=self.best.fitness
        self.trace[self.t,1]=self.avefitness
        print("Generation %d: optimal function value is: %f; average function value is %f" % (
            self.t, self.trace[self.t, 0], self.trace[self.t, 1]))
        while self.t < self.MAXGEN - 1:
            self.t += 1
            self.move()
            self.evaluate()
            best = np.max(self.fitness)
            bestIndex = np.argmax(self.fitness)
            if best>self.best.fitness:#更新最优点
                self.best = copy.deepcopy(self.population[bestIndex])
            self.avefitness = np.mean(self.fitness)
            self.trace[self.t, 0] = self.best.fitness
            self.trace[self.t, 1] = self.avefitness
            print("Generation %d: optimal function value is: %f; average function value is %f" % (
                self.t, self.trace[self.t, 0], self.trace[self.t, 1]))
        print("Optimal function value is: %f; " %
              self.trace[self.t, 0])
        print("Optimal solution is:")
        print(self.best.chrom)
        self.printResult()

# #测试单只萤火虫计算可否实现
# genee=[1,1,1,1,1,1,1]
# littleOne=FAIndividual(num_features,genee)
# littleOne.generate()
# littleOne.calculateFitness()
# littleOne.chrom=[True, True, True, True, True, True, True]
# print(littleOne.chrom)
# print(littleOne.fitness)


standred=[1,1,1,1,1,1,1,1]
vardim=num_features
fa=FireflyAlgorithm(sizepop,vardim,standred,10,[1.0,0.1,0.03])
fa.solve()




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

