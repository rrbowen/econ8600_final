import pandas
from sklearn.cluster import KMeans
import matplotlib.pyplot as pyplot
from sklearn.metrics import silhouette_score

#these are the results from the factor analysis code
data = pandas.read_csv("result.csv")

# print(data)

#turn the data back into a matrix 
data = data.values

# print(data)

pyplot.scatter(data[:,0], data[:,1])
pyplot.savefig("scatterplot_kmeans.png")


def run_kmeans(n, data):
	machine = KMeans(n_clusters=n)
	machine.fit(data)
	results = machine.predict(data)
	centroids = machine.cluster_centers_
	ssd = machine.inertia_
	silhouette = 0
	if n > 1:
		silhouette = silhouette_score(data, machine.labels_, metric='euclidean')
	pyplot.scatter(data[:,0], data[:,1], c=results)
	pyplot.scatter(centroids[:,0], centroids[:,1], c='red', marker="*", s=200)
	pyplot.savefig("scatterplot_color_" + str(n) + ".png")
	pyplot.close()
	return ssd, silhouette

# result = []
# for i in range(7):
# 	ssd = run_kmeans(i+1, data)	
# 	result.append(ssd)

# this is the way to do it
result= [ run_kmeans(i+1, data) for i in range(7)]

ssd_result = [ i[0] for i in result] 
silhouette_result = [ i[1] for i in result][1:]


pyplot.plot(range(1,8), ssd_result)
pyplot.savefig("ssd.png")
pyplot.close()

# # ssd_result_diff = []
# # for i,x  in enumerate(result):
# # 	diff = result[i-1] - x
# # 	ssd_result_diff.append(diff)

#find max silhoutte score by modifying the following line
ssd_result_diff = [ ssd_result[i-1] - x for i,x  in enumerate(ssd_result)][1:]

pyplot.plot(range(2,8), silhouette_result)
pyplot.savefig("silhouette_kmeans.png")
pyplot.close()


print("\nssd: \n", ssd_result)
print("\nssd differences: \n", ssd_result_diff)


print("\nsilhouette scores: \n", silhouette_result)
print("\nmax silhouette scores: \n", max(silhouette_result))
print("\nnumber of cluster with max silhouette scores: \n", 
	silhouette_result.index(max(silhouette_result))+2)


