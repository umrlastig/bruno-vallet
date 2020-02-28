# input/output
img_filename = "lena.jpg"
filt_img_filename = "lena_filt.jpg"
mask_img_filename = "lenaMask.png"
l = 0.001
smooth = 0

print('Initialisation')
import numpy as np
import scipy as sp
import scipy.misc as sm
import scipy.sparse.linalg as ssl

def Gc(nl, nc):
    ret = np.zeros(((nc-1)*nl,nc*nl))
    i_l=0
    for l in range(nl):
        for c in range(nc-1):
            ret[i_l, c + 1 + nc * l] = 1
            ret[i_l, c + nc * l] = -1
            i_l+=1
    return ret

def Gl(nl, nc):
    ret = np.zeros(((nl-1)*nc,nc*nl))
    i_l=0
    for l in range(nl-1):
        for c in range(nc):
            ret[i_l, c + nc * (l+1)] = 1
            ret[i_l, c + nc * l] = -1
            i_l+=1
    return ret

def Gcs(nl, nc):
    data = []
    row_ind = []
    col_ind = []
    i_l=0
    for l in range(nl):
        for c in range(nc-1):
            data.append(-1)
            row_ind.append(i_l)
            col_ind.append(c + nc * l)
            data.append(1)
            row_ind.append(i_l)
            col_ind.append(c + 1 + nc * l)
            i_l+=1
    return sp.sparse.csr_matrix((data, (row_ind, col_ind)))

def Gls(nl, nc):
    data = []
    row_ind = []
    col_ind = []
    i_l=0
    for l in range(nl-1):
        for c in range(nc):
            data.append(-1)
            row_ind.append(i_l)
            col_ind.append(c + nc * l)
            data.append(1)
            row_ind.append(i_l)
            col_ind.append(c + nc * (l+1))
            i_l+=1
    return sp.sparse.csr_matrix((data, (row_ind, col_ind)))

def sparse_eye(n):
    return sp.sparse.csr_matrix((np.ones(n), (range(n), range(n))))

def sparse_diag(v):
    n=v.shape[0]
    return sp.sparse.csr_matrix((v, (range(n), range(n))))

print('Reading %s' % img_filename)
image = sm.imread(img_filename).astype(float)
print(image.shape)
#image = image[:10,:10]

grad_c = Gcs(image.shape[0],image.shape[1])
grad_l = Gls(image.shape[0],image.shape[1])

G = np.dot(grad_c.transpose(),grad_c)+np.dot(grad_l.transpose(),grad_l)
f = np.reshape(image, image.shape[0]*image.shape[1])

if smooth:
    A = sparse_eye(G.shape[0])+l*G
    u = ssl.spsolve(A, f)
    img_filt = np.reshape(u, (image.shape[0],image.shape[1])).astype(uint8)

else:
    # mask
    print('Reading %s' % mask_img_filename)
    mask = sm.imread(mask_img_filename)
    print(mask.shape)
    mask = mask[:image.shape[0], :image.shape[1]].astype(float)/255
    mask_v = np.reshape(mask, mask.shape[0] * mask.shape[1])
    M = sparse_diag(mask_v)
    A = M + l*G
    b = M.dot(f)
    u = ssl.spsolve(A, b)
    img_filt = np.reshape(u, (image.shape[0],image.shape[1])).astype(np.uint8)
    
print('Saving %s' % filt_img_filename)
sm.imsave(filt_img_filename, img_filt)
