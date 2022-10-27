# input/output
# basename = "lena.jpg"
basename = 'contour4'
img_filename = basename + ".png"
mask_img_filename = basename + "_mask.png"
filt_img_filename = basename + "_filt.png"
bin_img_filename = basename + "_bin.png"

l = 1.000001
eps = 0.0000001
mode = 'contour'  # 'smooth' 'hole_fill' 'contour'

print('Initialisation')
import numpy as np
import scipy as sp
# import scipy.misc as sm
import scipy.sparse.linalg as ssl
import imageio


def Gc(nl, nc):
    ret = np.zeros(((nc - 1) * nl, nc * nl))
    i_l = 0
    for l in range(nl):
        for c in range(nc - 1):
            ret[i_l, c + 1 + nc * l] = 1
            ret[i_l, c + nc * l] = -1
            i_l += 1
    return ret


def Gl(nl, nc):
    ret = np.zeros(((nl - 1) * nc, nc * nl))
    i_l = 0
    for l in range(nl - 1):
        for c in range(nc):
            ret[i_l, c + nc * (l + 1)] = 1
            ret[i_l, c + nc * l] = -1
            i_l += 1
    return ret


def Gcs(nl, nc):
    data = []
    row_ind = []
    col_ind = []
    i_l = 0
    for l in range(nl):
        for c in range(nc - 1):
            data.append(-1)
            row_ind.append(i_l)
            col_ind.append(c + nc * l)
            data.append(1)
            row_ind.append(i_l)
            col_ind.append(c + 1 + nc * l)
            i_l += 1
    return sp.sparse.csr_matrix((data, (row_ind, col_ind)))


def Gls(nl, nc):
    data = []
    row_ind = []
    col_ind = []
    i_l = 0
    for l in range(nl - 1):
        for c in range(nc):
            data.append(-1)
            row_ind.append(i_l)
            col_ind.append(c + nc * l)
            data.append(1)
            row_ind.append(i_l)
            col_ind.append(c + nc * (l + 1))
            i_l += 1
    return sp.sparse.csr_matrix((data, (row_ind, col_ind)))


def sparse_eye(n):
    return sp.sparse.csr_matrix((np.ones(n), (range(n), range(n))))


def sparse_diag(v):
    n = v.shape[0]
    return sp.sparse.csr_matrix((v, (range(n), range(n))))


def sparse_select(vect):
    whr = np.where(vect)[0]
    print(whr)
    nc = whr.shape[0]
    return sp.sparse.csr_matrix((np.ones(nc), (range(nc), whr)), shape=(nc, vect.shape[0]))


def contours(image, mask, lbda, eps):
    n = image.shape[0] * image.shape[1]
    mask_v = np.reshape(mask, n)
    Ex = sparse_select(np.logical_and(mask_v != 0, mask_v != 255))
    E1 = sparse_select(mask_v == 255)
    imagec = image[:image.shape[0], :image.shape[1] - 1]
    imagel = image[:image.shape[0] - 1, :image.shape[1]]
    nc = imagec.shape[0] * imagec.shape[1]
    nl = imagel.shape[0] * imagel.shape[1]
    fc = np.reshape(imagec, nc)
    fl = np.reshape(imagel, nl)
    Dc = lbda * sparse_eye(nc) - sparse_diag(fc)
    Dl = lbda * sparse_eye(nl) - sparse_diag(fl)
    M = np.dot(grad_c.transpose(), np.dot(Dc, grad_c)) + np.dot(grad_l.transpose(), np.dot(Dl, grad_l))
    Mx = np.dot(Ex, M)
    Mxx = np.dot(Mx, Ex.transpose())
    Mx1 = Mx.dot(E1.transpose())
    u1 = np.ones(E1.shape[0])
    b = Mx1.dot(u1)
    ux = -ssl.spsolve(Mxx+eps*sparse_eye(Mxx.shape[0]), b)
    u = Ex.transpose().dot(ux) + E1.transpose().dot(u1)
    grad_cu = grad_c.dot(u)
    print(np.max(grad_cu))
    imgrad_cu2 = np.reshape(grad_cu*grad_cu, (image.shape[0], image.shape[1]-1))
    print(np.max(imgrad_cu2))
    grad_lu = grad_l.dot(u)
    print(np.max(grad_lu))
    imgrad_lu2 = np.reshape(grad_lu * grad_lu, (image.shape[0]-1, image.shape[1]))
    print(np.max(imgrad_lu2))
    imgrad2 = np.zeros(image.shape)
    imgrad2[:image.shape[0], :image.shape[1] - 1] += imgrad_cu2
    print(np.max(imgrad2))
    imgrad2[:image.shape[0]-1, :image.shape[1]] += imgrad_lu2
    print(np.max(imgrad2))
    img_filt = np.reshape(255 * u, (image.shape[0], image.shape[1])).astype(np.uint8)
    return img_filt, imgrad2


print('Reading %s' % img_filename)
image = imageio.imread(img_filename).astype(float) / 255
print(image.shape)
# image = image[:10,:10]

grad_c = Gcs(image.shape[0], image.shape[1])
grad_l = Gls(image.shape[0], image.shape[1])

f = np.reshape(image, image.shape[0] * image.shape[1])

if mode == 'smooth':
    G = np.dot(grad_c.transpose(), grad_c) + np.dot(grad_l.transpose(), grad_l)
    A = sparse_eye(G.shape[0]) + l * G
    u = ssl.spsolve(A, f)
    img_filt = np.reshape(u * 255, (image.shape[0], image.shape[1])).astype(np.uint8)

elif mode == 'hole_fill':
    # mask
    print('Reading %s' % mask_img_filename)
    mask = imageio.imread(mask_img_filename)
    print(mask.shape)
    mask = mask[:image.shape[0], :image.shape[1]].astype(float) / 255
    mask_v = np.reshape(mask, mask.shape[0] * mask.shape[1])
    M = sparse_diag(mask_v)
    G = np.dot(grad_c.transpose(), grad_c) + np.dot(grad_l.transpose(), grad_l)
    A = M + l * G
    b = M.dot(f)
    u = ssl.spsolve(A, b)
    img_filt = np.reshape(255 * u, (image.shape[0], image.shape[1])).astype(np.uint8)

elif mode == 'contour':
    # mask
    print('Reading %s' % mask_img_filename)
    mask = imageio.imread(mask_img_filename)
    print(mask.shape)
    mask = mask[:image.shape[0], :image.shape[1]]
    for i in range(10):
        img_filt, imgrad2 = contours(image, mask, l, eps)
        #mask[np.where(img_filt < 64)] = 0
        #mask[np.where(img_filt > 210)] = 255
        imageio.imsave('mask%s.png'%i, mask)
        print(np.max(imgrad2))
        imageio.imsave('grad%s.png' % i, imgrad2) # 255*imgrad2.astype(np.uint8))
        image = image + 10 * imgrad2
        image = np.minimum(np.ones(image.shape), image)
        imageio.imsave('image%s.png' % i, image)

print('Saving %s' % filt_img_filename)
imageio.imsave(filt_img_filename, img_filt)
imageio.imsave(bin_img_filename, 255 * (img_filt > 128))
