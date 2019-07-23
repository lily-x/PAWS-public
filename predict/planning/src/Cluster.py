# Author: Gael Varoquaux <gael.varoquaux@normalesup.org>, Brian Cheung
# License: BSD 3 clause

import time

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

from sklearn.feature_extraction import image
from sklearn.cluster import spectral_clustering


def cluster(face):
    # Convert the image into a graph with the value of the gradient on the
    # edges.
    graph = image.img_to_graph(face)
    
    # Take a decreasing function of the gradient: an exponential
    # The smaller beta is, the more independent the segmentation is of the
    # actual image. For beta=1, the segmentation is close to a voronoi
    beta = 5
    eps = 1e-6
    graph.data = np.exp(-beta * graph.data / graph.data.std()) + eps
    
    # Apply spectral clustering (this step goes much faster if you have pyamg
    # installed)
    N_REGIONS = 25
    
    for assign_labels in ('kmeans', 'discretize'):
        t0 = time.time()
        labels = spectral_clustering(graph, n_clusters=N_REGIONS,
                                     assign_labels=assign_labels, random_state=1)
        t1 = time.time()
        labels = labels.reshape(face.shape)
    
        plt.figure(figsize=(5, 5))
        plt.imshow(face)
        for l in range(N_REGIONS):
            plt.contour(labels == l, contours=1,
                       )
        plt.xticks(())
        plt.yticks(())
        title = 'Spectral clustering: %s, %.2fs' % (assign_labels, (t1 - t0))
        print(title)
        plt.title(title)
    plt.show()