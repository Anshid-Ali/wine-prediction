from sklearn.datasets import load_wine
data = load_wine()

x=data.data
y=data.target

print(x.shape)
print(y.shape)


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=3)
print(knn)
knn.fit(X, y)
knn.predict([[12.86,1.35,2.32,18,122,1.51,1.25,.21,.94,4.1,.76,1.29,630
]])