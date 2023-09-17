import csv
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram
from sklearn.manifold import MDS

def main():
    data = pd.read_csv('files/customers.csv', delimiter=';')
    selected_features = data[['gender', 'age', 'tv_channel', 'profession', 'press']]
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(selected_features)

    distance_matrix = linkage(scaled_data, method='ward', metric='euclidean')
    plt.figure(figsize=(10, 5))
    dendrogram(distance_matrix)
    plt.show()

    kmeans = KMeans(n_clusters=3, random_state=0)
    kmeans.fit(scaled_data)
    labels = kmeans.labels_

    mds = MDS(n_components=2, dissimilarity='euclidean')
    X_mds = mds.fit_transform(scaled_data)

    # Plot clusters
    plt.figure(figsize=(10, 5))
    plt.scatter(X_mds[:, 0], X_mds[:, 1], c=labels)
    plt.show()


if __name__ == '__main__':
    main()

