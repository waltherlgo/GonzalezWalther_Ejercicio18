import glob
import numpy as np
import matplotlib.pyplot as plt
import sklearn.cluster
files_val = glob.glob('./Imagenes/*.png')
Data_val=np.zeros((87,30000))
for i,names in enumerate(files_val):
    data = (plt.imread(names)).astype(float)
    X = data.reshape((-1,1))
    Data_val[i,:]=X.T
Inerlist=[]
for n_clusters in range (1,21):
    k_means = sklearn.cluster.KMeans(n_clusters=n_clusters)
    k_means.fit(Data_val)
    Inerlist.append(k_means.inertia_)
plt.plot(np.arange(1,21),Inerlist)
plt.savefig("inercia.png")
mejorclust=4
k_means = sklearn.cluster.KMeans(n_clusters=mejorclust)
k_means.fit(Data_val)
cluster=k_means.predict(Data_val)
ars=np.zeros((4,5))
for i in range(4):
    center=k_means.cluster_centers_[i]
    dist=np.zeros(87)
    for im in range(87):
        dist[im]=np.linalg.norm(Data_val[im,:]-center)
    ars[i,:]=(np.argsort(dist))[:5]
ars=ars.astype(int)
plt.figure(figsize=(30,20))
for i in range(4):
    for i2 in range(5):
        data = plt.imread(files_val[ars[i][i2]])
        plt.subplot(4,5,i*5+i2+1)
        plt.imshow(data)
        plt.title("K"+str(i))
plt.savefig("ejemplo_clases.png")