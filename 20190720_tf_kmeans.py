import tensorflow as tf
import numpy as np
import time
import matplotlib
import matplotlib.pyplot as plt

from sklean.datasets.samples_generator import make_blobs
from sklean.datasets.samples_generator import make_circles

DATA_TYPE='blobs'
if DATA_TYPE=='circles':
    K=2
else:
    K=4
    
MAX_ITERS=1000

start=time.time()

centers=[(-2,-2),(-2,1.5),(1.5,-2),(2,1.5)]
if DATA_TYPE=='circles':
    data,features=make_circles(n_samples=200,shuffle=True,noise=0.01,factor=0.4)
else:
    data,features=make_blobs(n_samples=200,centers=centers,n_features=2,cluster_std=0.8,shuffle=False,random_state=42)
    
fig,ax=plt.subplots()
ax.scatter(np.asarray(centers).transpose()[0],np.asarray(centers).transpose()[1],marker='o',s=250)
plt.show()

fig,ax=plt.subplots()
if DATA_TYPE=='blobs':
    ax.scatter(np.asarray(centers).transpose()[0],np.asarray(centers).transpose()[1],marker='o',s=250)
    ax.scatter(data.transpose()[0],data.transpose()[1],marker='o',s=100,c=features,cmap=plt.cm.coolwarm)
    plt.plot()
    
points=tf.Variable(data)
cluster_assignments=tf.Variable(tf.zeros([N],dtype=tf.int64))
centroids=tf.Variable(tf.slice(points.initialized_value(),[0,0],[K,2]))

sess=tf.Session()
sess.run(tf.initialize_all_variables())
sess.run(centroids)
rep_centroids=tf.reshape(tf.tile(centroids,[N,1]),[N,K,2])
rep_points=tf.reshape(tf.tile(points,[1,K]),[N,K,2])
sum_squares=tf.reduce_sum(tf.square(rep_points-rep_centroids),reduction_indices=2)
