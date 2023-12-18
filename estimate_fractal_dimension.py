# By Jackson Thissell
# 12/11/2023

from sklearn.metrics.pairwise import euclidean_distances
from sklearn.linear_model import LinearRegression
import numpy as np
import math


def estimate_fractal_dimension(pts, steps=100):
    # Get distances between all points
    dists = euclidean_distances(pts, pts)

    # Get the upper triangular portion since
    # dist(x, y) = dist(y, x), and then flatten them
    dists = np.triu(dists).flatten()

    # Remove all zeroes
    dists = dists[dists != 0]

    # Get minimum and mean
    d_min = dists.min()
    d_avg = dists.mean()

    corr_int = []
    for i in range(1, steps + 1):
        # Take various epsilon samplings at logarithmic rate
        eps = d_min + (d_avg - d_min) / i

        # Get correlation integral for epsilon sample
        g = len(np.where(dists < eps)[0])
        corr_int.append([math.log(eps), math.log(g/len(pts)**2)])

    # Fit linear approximation to log-log plot
    x, y = np.asarray(corr_int).T
    reg = LinearRegression().fit(x.reshape(-1, 1), y)

    # The slope approximates the dimension
    return reg.coef_