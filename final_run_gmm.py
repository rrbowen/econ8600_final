import pandas
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

import matplotlib.pyplot as pyplot

from sklearn.metrics import silhouette_score

data = pandas.read_csv("result.csv")

print(data)

data = data.values

pyplot.scatter(data[:,1], data[:,2])
pyplot.savefig("scatter_gmm.png")
pyplot.close()

#data = data[:,1:3]

#print(data)


def run_gmm(n, data):
	gmm_machine = GaussianMixture(n_components=n)
	gmm_results = gmm_machine.fit_predict(data)
	silhouette = 0
	if n > 1:
		silhouette = silhouette_score(data, gmm_results, metric = 'euclidean')
	pyplot.scatter(data[:,0], data[:,1], c=gmm_results)
	pyplot.savefig("scatter_gmm_" + str(n) + ".png")
	pyplot.close()
	return silhouette


gmm_silhouette_list = [ run_gmm(i+1, data) for i in range(7)]
print(gmm_silhouette_list)

