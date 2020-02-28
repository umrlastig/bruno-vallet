# input/output
img_filename = "hyst.png"
sigma = 1.

print('Initialisation')
import numpy as np
import scipy as sp
import scipy.misc as sm
import skimage as sf
import random
import math 

def Coords(img,seuil):
    ret = []
    for l in range(1,img.shape[0]):
        for c in range(1,img.shape[1]):
            if img[l,c,0]>seuil:
                ret.append(np.array([l,c]))
    return ret

def DroiteL2(coords):
    G = np.zeros(2)
    n_pt = 0
    for lc in coords:
        G += lc
        n_pt += 1
    G /= n_pt
    E = np.zeros((2,2))
    for lc in coords:
        v = lc-G
        E = E + np.outer(v,v)
    evals, evects = np.linalg.eig(E)
    d = evects[:,0]
    if evals[1] > evals[0]:
        d = evects[:,1]
    sG = G.dot(d)
    s_min = s_max = sG
    for lc in coords:
        s = lc.dot(d)
        if s < s_min:
            s_min = s
        if s > s_max:
            s_max = s
    return G + (s_min-sG)*d, G + (s_max-sG)*d

def normal(P1, P2):
    xf = (P2-P1).astype(float)
    d = xf / math.sqrt(float(xf.dot(xf)))
    return np.array([d[1],-d[0]])

def Ransac(coords, n_iter, seuil):
    n_pts = len(coords)
    print('%d contour points'%n_pts)
    n_in_max = 0
    P1_max = P2_max = coords[0]
    for iter in range(n_iter):
        P1 = coords[random.randint(0,n_pts-1)]
        P2 = coords[random.randint(0,n_pts-1)]
        n = normal(P2, P1)
        n_in = 0
        for x in coords:
            if abs(n.dot(x-P1))<seuil:
                n_in += 1
        if n_in > n_in_max:
            n_in_max = n_in
            P1_max = P1
            P2_max = P2
    print('Best line has %d inliers'%n_in_max)
    inliers = []
    outliers = []
    n = normal(P2_max, P1_max)
    for x in coords:
        if abs(n.dot(x-P1_max)) < seuil:
            inliers.append(x)
        else:
            outliers.append(x)
    return inliers, outliers
    
def MultiRansac(coords, n_iter, seuil, n_in_min):
    ret = []
    outliers = coords
    while True:
        inliers, outliers = Ransac(outliers, n_iter, seuil)
        if len(inliers) < n_in_min:
            return ret
        ret.append(DroiteL2(inliers))
    
print('Reading %s' % img_filename)
img = sm.imread(img_filename)
P_min, P_max = DroiteL2(Coords(img,250))
droites = MultiRansac(Coords(img,100),100,2,150)
print('RANSAC found %d lines'%len(droites))

from PIL import Image, ImageDraw
im = Image.open(img_filename)
d = ImageDraw.Draw(im)
d.line([(P_min[1],P_min[0]), (P_max[1],P_max[0])], fill=(255, 255, 0))

for droite in droites:
    d.line([(droite[0][1],droite[0][0]), (droite[1][1],droite[1][0])],
        fill=(0, 255, 0))

im.save("ransac.png")

