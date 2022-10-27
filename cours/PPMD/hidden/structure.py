# input
img_filename = "hyst.png"
n_iter=100
seuil=1
n_in_min=150

print('Initialisation')
import numpy as np
import scipy as sp
import random
import math
import imageio


def Coords(img, seuil):
    ret = []
    for l in range(1, img.shape[0]):
        for c in range(1, img.shape[1]):
            if img[l, c, 0] > seuil:
                ret.append(np.array([l, c]))
    return ret


def DroiteL2(coords):
    G = np.zeros(2)
    n_pt = 0
    for lc in coords:
        G += lc
        n_pt += 1
    G /= n_pt
    print('G=')
    print(G)
    E = np.zeros((2, 2))
    for lc in coords:
        v = lc - G
        E = E + np.outer(v, v)
    print('El2=')
    print(E)
    evals, evects = np.linalg.eig(E)
    print('Valeurs propres:')
    print(evals)
    print('Vecteurs propres:')
    print(evects)
    d = evects[:, 0]
    n = evects[:, 1]
    if evals[1] > evals[0]:
        d = evects[:, 1]
        n = evects[:, 0]
    print('d:')
    print(d)
    print('n:')
    print(n)
    sG = G.dot(d)
    s_min = s_max = sG
    for lc in coords:
        s = lc.dot(d)
        if s < s_min:
            s_min = s
        if s > s_max:
            s_max = s
    return G + (s_min - sG) * d, G + (s_max - sG) * d, G, n


def normal(P1, P2):
    xf = (P2 - P1).astype(float)
    d = xf / math.sqrt(float(xf.dot(xf)))
    return np.array([d[1], -d[0]])


def Ransac(coords):
    n_pts = len(coords)
    print('%d contour points' % n_pts)
    n_in_max = 0
    P1_max = P2_max = coords[0]
    for iter in range(n_iter):
        P1 = coords[random.randint(0, n_pts - 1)]
        P2 = coords[random.randint(0, n_pts - 1)]
        n = normal(P2, P1)
        n_in = 0
        for x in coords:
            if abs(n.dot(x - P1)) < seuil:
                n_in += 1
        if n_in > n_in_max:
            n_in_max = n_in
            P1_max = P1
            P2_max = P2
    print('Best line has %d inliers' % n_in_max)
    inliers = []
    outliers = []
    n = normal(P2_max, P1_max)
    for x in coords:
        if abs(n.dot(x - P1_max)) < seuil:
            inliers.append(x)
        else:
            outliers.append(x)
    return inliers, outliers


def MultiRansac(coords):
    ret = []
    outliers = coords
    iter=0
    while True:
        iter += 1
        print("RANSAC iteration %d" % iter)
        inliers, outliers = Ransac(outliers)
        if len(inliers) < n_in_min:
            return ret
        ret.append(DroiteL2(inliers))


print('Reading %s' % img_filename)
img = imageio.imread(img_filename)
P_min, P_max, G, n = DroiteL2(Coords(img, 250))
print('Pmin/max:')
print(P_min)
print(P_max)
droites = MultiRansac(Coords(img, 100))
print('RANSAC found %d lines' % len(droites))

import matplotlib.pyplot as plt

plt.figure()
plt.imshow(img)
plt.plot((P_min[1], P_max[1]), (P_min[0], P_max[0]), color=(0, 1, 0))
Gp = G + 50*n
plt.plot((G[1], Gp[1]), (G[0], Gp[0]), color=(1, 0, 0))
for droite in droites:
    plt.plot((droite[0][1], droite[1][1]), (droite[0][0], droite[1][0]), color=(1, 0, 1))
plt.title("image")
plt.show()
