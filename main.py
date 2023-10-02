import pandas as pd
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import linkage
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram
from sklearn.manifold import MDS


def main():
    data = pd.read_csv('files/customers.csv', delimiter=';')



    distance_matrix = linkage(data, method='ward', metric='euclidean')
    plt.figure(figsize=(10, 5))
    dendrogram(distance_matrix, p=3, truncate_mode='level')
    plt.title('Дендограмма')
    plt.show()

    kmeans = KMeans(n_clusters=2, random_state=0)
    kmeans.fit(data)
    labels = kmeans.labels_

    mds = MDS(n_components=2, dissimilarity='euclidean', random_state=0)
    X_mds = mds.fit_transform(data)

    plt.figure(figsize=(10, 5))
    plt.scatter(X_mds[:, 0], X_mds[:, 1], c=labels)
    plt.title('Многомерное шкалирование')
    plt.show()

    sums = []

    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, random_state=0)
        kmeans.fit(data)
        sums.append(kmeans.inertia_)

    plt.figure(figsize=(10, 5))
    plt.plot(range(1, 11), sums, marker='o', linestyle='--')
    plt.title('График "каменистая осыпь"')
    plt.xlabel('Количество кластеров')
    plt.ylabel('Сумма квадратов расстояний')
    plt.show()

    kmeans = KMeans(n_clusters=2, random_state=0)
    kmeans.fit(data)
    data['cluster'] = kmeans.labels_

    print(data.groupby('cluster').mean())
    print(data.groupby('cluster').size())


if __name__ == '__main__':
    main()
