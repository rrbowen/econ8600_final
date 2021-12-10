import pandas
import numpy
from factor_analyzer import FactorAnalyzer
import matplotlib.pyplot as pyplot


dataset = pandas.read_csv("dataset_final.csv")
print(dataset)
print(dataset.shape)

dataset.drop(['country'], axis=1, inplace = True)
dataset.drop(['Unnamed: 0'],axis=1, inplace = True)
dataset.dropna(inplace = True)


print(dataset)
print(dataset.shape)