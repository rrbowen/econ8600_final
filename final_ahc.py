import pandas
import matplotlib.pyplot as pyplot
import scipy.cluster.hierarchy as shc 
from sklearn.cluster import AgglomerativeClustering

dataset = pandas.read_csv("result_headings.csv")

print(dataset)

#of no use
#pyplot.scatter(dataset['x1'], dataset['x2'])
#pyplot.savefig("scatterplot_ahc.png")
#pyplot.close()

#from the internet.  Error message was "RecursionError: maximum recursion depth exceeded while getting the str of an object"
import sys
sys.setrecursionlimit(5000)

pyplot.title("Dendrogram")
dendrogram_object = shc.dendrogram(shc.linkage(dataset, method="complete"))
pyplot.savefig("dendrogram.png")
pyplot.close()

machine = AgglomerativeClustering(n_clusters=2, affinity="euclidean", linkage="ward")
results_ahc = machine.fit_predict(dataset)

pyplot.scatter(dataset['x1'], dataset['x2'], c=results_ahc)
pyplot.savefig("scatterplot_ahc_color.png")
pyplot.close()




