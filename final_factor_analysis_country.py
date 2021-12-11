import pandas
import numpy
from factor_analyzer import FactorAnalyzer
import matplotlib.pyplot as pyplot

dataset = pandas.read_csv("dataset_final.csv")
print(dataset)
print(dataset.shape)

#dataset.drop(['country'], axis=1, inplace = True)
dataset.drop(['Unnamed: 0'],axis=1, inplace = True)
dataset.dropna(inplace = True)

#print(dataset)
#print(dataset.shape)

machine = FactorAnalyzer(n_factors=41, rotation = None)
machine.fit(dataset)
ev, v = machine.get_eigenvalues()
#print(ev)

pyplot.scatter(range(1,dataset.shape[1]+1),ev)
pyplot.savefig("plot.png")
pyplot.close()

machine = FactorAnalyzer(n_factors=3, rotation = 'Varimax')
machine.fit(dataset)
loadings = machine.loadings_
#numpy.set_printoptions(suppress=True)

#print("\nfactor loadings:\n")
#print(loadings)
#print(machine.get_factor_variance)

#turn into an array
dataset = dataset.values

result = numpy.dot(dataset, loadings)

print(result)
print(result.shape)

#"result" is the dataset we will use for clustering.  It does not include country for now

from numpy import asarray
from numpy import savetxt
savetxt('result.csv', result, delimiter=',')


