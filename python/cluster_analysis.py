

# Cluster analysis


# import some libraries

import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


# define some useful functions (from Hands-On-ML2)

def plot_data(X):
    plt.plot(X[:, 0], X[:, 1], 'k.', markersize=2)

def plot_centroids(centroids, weights=None, circle_color='w', cross_color='k'):
    if weights is not None:
        centroids = centroids[weights > weights.max() / 10]
    plt.scatter(centroids[:, 0], centroids[:, 1],
                marker='o', s=5, linewidths=4,
                color=circle_color, zorder=10, alpha=0.9)
    plt.scatter(centroids[:, 0], centroids[:, 1],
                marker='x', s=2, linewidths=12,
                color=cross_color, zorder=11, alpha=1)

def plot_decision_boundaries(clusterer, X, resolution=1000, show_centroids=True,
                             show_xlabels=True, show_ylabels=True):
    mins = X.min(axis=0) - 0.1
    maxs = X.max(axis=0) + 0.1
    xx, yy = np.meshgrid(np.linspace(mins[0], maxs[0], resolution),
                         np.linspace(mins[1], maxs[1], resolution))
    Z = clusterer.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(Z, extent=(mins[0], maxs[0], mins[1], maxs[1]),
                cmap="Pastel2")
    plt.contour(Z, extent=(mins[0], maxs[0], mins[1], maxs[1]),
                linewidths=1, colors='k')
    plot_data(X)
    if show_centroids:
        plot_centroids(clusterer.cluster_centers_)


# read in data

f = open('cluster_raw.dat', 'r')

data = np.loadtxt(f, skiprows=1, delimiter=None)

f.close()


# examine raw data


# print min/max
for i in range(5):
    print ("x"+str(i)+" = ", min(data[:,i+1]), max(data[:,i+1]))


# standardize data: important data preprocessing step, but also depends on data

normalized = StandardScaler()
data1 = normalized.fit_transform(data)


# plot raw data

fig = plt.figure(figsize=(12, 8))

rows = 5
columns = 2

grid = plt.GridSpec(rows, columns, wspace = 0.20, hspace = 0)

for i in range(5):
    plt.subplot(grid[i,0])
    plt.scatter(data[:,0], data[:,i+1], s=1)
    if (i == 0):
       plt.title("Raw data", fontsize=14)
    plt.annotate(r"$x_{:d}$".format(i), xy=(0.02, 0.8), 
                 xycoords="axes fraction", fontsize=14)

    plt.subplot(grid[i,1])
    plt.scatter(data1[:,0], data1[:,i+1], s=1)
    if (i == 0):
       plt.title("Standardized data", fontsize=14)
    plt.annotate(r"$x_{:d}$".format(i), xy=(0.02, 0.8), 
                 xycoords="axes fraction", fontsize=14)

plt.savefig("../figs/cluster_raw_data.png", bbox_inches='tight')

plt.show()


# perform cluster analysis

data = data1

fig = plt.figure(figsize=(10, 10))

rows = 5
columns = 3

grid = plt.GridSpec(rows, columns, wspace = 0.25, hspace = 0.25)


# number of clusters to try 
nk = 10


# save inertia and silhouette_score for plotting 

# number of data pairs
nx = 15

inertiax          = np.zeros([nx, nk])
silhouette_scorex = np.zeros([nx, nk])


n = 0
for i in range(0,6):
    for j in range(i+1,6):
        X = data[:,[i,j]]
        kmeans_per_k = [KMeans(n_clusters=k, random_state=42).fit(X)
                        for k in range(1, nk)]

        inertias = [model.inertia_ for model in kmeans_per_k]
        inertiax[n,1:10] = inertias

        silhouette_scores = [silhouette_score(X, model.labels_)
                             for model in kmeans_per_k[1:]]
        silhouette_scorex[n,2:10] = silhouette_scores


        # plot clusters
        plt.subplot(grid[n//3,n%3])
        if (n == 1 or n == 6 or n == 10 or n == 14):
           plot_decision_boundaries(kmeans_per_k[1], X)
        elif (n == 0 or n == 4):
           plot_decision_boundaries(kmeans_per_k[3], X)
        else:
           plot_decision_boundaries(kmeans_per_k[2], X)

        if (i == 0):
           plt.annotate(r"($x$"+"|"+r"$x_{:d}$)".format(j-1), 
                        xy=(0.70, 0.75), xycoords="axes fraction", fontsize=14)
        else: 
           plt.annotate(r"($x_{:d}$".format(i-1)+"|"+r"$x_{:d}$)".format(j-1), 
                        xy=(0.70, 0.75), xycoords="axes fraction", fontsize=14)

        n = n + 1


plt.suptitle("Clusters", fontsize=16)

plt.savefig("../figs/clusters.png", bbox_inches='tight')

plt.show()



# plot inertia

fig = plt.figure(figsize=(12, 8))

rows = 5
columns = 3

grid = plt.GridSpec(rows, columns, wspace = 0.25, hspace = 0)


n = 0
for i in range(0,6):
    for j in range(i+1,6):
        plt.subplot(grid[n//3,n%3])
        plt.plot(range(1, 10), inertiax[n,1:10], marker=".", markersize=10)
        plt.xticks(np.arange(1, 10))
        plt.xlabel("$k$", fontsize=14)

        if (i == 0):
           plt.annotate(r"($x$"+"|"+r"$x_{:d}$)".format(j-1),
                        xy=(0.70, 0.75), xycoords="axes fraction", fontsize=14)
        else:
           plt.annotate(r"($x_{:d}$".format(i-1)+"|"+r"$x_{:d}$)".format(j-1),
                        xy=(0.70, 0.75), xycoords="axes fraction", fontsize=14)

        n = n + 1


plt.suptitle("Inertia", fontsize=16)

plt.savefig("../figs/inertia_vs_k.png", bbox_inches='tight')


plt.show()


# plot silhouette score

fig = plt.figure(figsize=(12, 8))

rows = 5
columns = 3

grid = plt.GridSpec(rows, columns, wspace = 0.25, hspace = 0)


n = 0
for i in range(0,6):
    for j in range(i+1,6):
        plt.subplot(grid[n//3,n%3])
        plt.plot(range(2, 10), silhouette_scorex[n,2:10], marker=".", markersize=10)
        plt.xticks(np.arange(2, 10))
        plt.xlabel("$k$", fontsize=14)

        if (i == 0):
           plt.annotate(r"($x$"+"|"+r"$x_{:d}$)".format(j-1), 
                        xy=(0.70, 0.75), xycoords="axes fraction", fontsize=14)
        else: 
           plt.annotate(r"($x_{:d}$".format(i-1)+"|"+r"$x_{:d}$)".format(j-1), 
                        xy=(0.70, 0.75), xycoords="axes fraction", fontsize=14)

        n = n + 1


plt.suptitle("Silhouette score", fontsize=16)

plt.savefig("../figs/silhouette_score_vs_k.png", bbox_inches='tight')


plt.show()


