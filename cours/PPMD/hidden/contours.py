# input/output
img_filename = "batG.png"
sigma = 1.
seuil_bas = 0.02
seuil_haut = 0.1
canny_filename = 'batG.cannyH.png'
lapl_filename = 'batG.laplH.png'

print('Initialisation')
import numpy as np
import scipy as sp
# import scipy.misc as sm
import skimage as sf
import skimage.filters as filters
import imageio
from scipy import ndimage
import matplotlib.pyplot as plt


def snap(A):
    return (A > 1).astype(int) - (A < -1).astype(int)


def Cluster(mask):
    processed = np.zeros(image.shape, dtype=np.bool)
    ret = []
    for l in range(1, image.shape[0] - 1):
        for c in range(1, image.shape[1] - 1):
            seed = (l, c)
            if not processed[seed] and mask[seed]:
                cluster = []
                pile = [seed]
                processed[seed] = 1
                while len(pile) > 0:
                    pix = pile[-1]
                    pile.pop()
                    cluster.append(pix)
                    nn = 0
                    for dl in range(-1, 2):
                        for dc in range(-1, 2):
                            new_pix = (pix[0] + dl, pix[1] + dc)
                            if not processed[new_pix] and mask[new_pix]:
                                pile.append(new_pix)
                                processed[new_pix] = 1
                                nn += 1
                ret.append(cluster)
    return ret


def Hyst(mask, grad_norm, seuil_haut):
    clusters = Cluster(mask)
    ret = np.zeros(image.shape, dtype=np.bool)
    for cluster in clusters:
        ok = 0
        for pix in cluster:
            if grad_norm[pix] > seuil_haut:
                ok = 1
        if ok:
            for pix in cluster:
                ret[pix] = 1
    return ret


def ContoursCanny(image, sigma, seuil_bas, seuil_haut):
    # gauss_img = sf.filter.gaussian_filter(image, sigma=sigma, mode='nearest').astype(float)
    gauss_img = filters.gaussian(image, sigma=sigma, mode='nearest', preserve_range=True)
    grad_c = np.array([[-1, 1]])
    grad_l = grad_c.transpose()
    grad_c_img = sp.ndimage.convolve(gauss_img, grad_c)
    imageio.imwrite(img_filename + '.gc.jpg', grad_c_img)
    grad_l_img = sp.ndimage.convolve(gauss_img, grad_l)
    imageio.imwrite(img_filename + '.gl.jpg', grad_l_img)
    grad_norm = np.sqrt(
        np.multiply(grad_c_img, grad_c_img) +
        np.multiply(grad_l_img, grad_l_img))
    imageio.imwrite(img_filename + '.gnorm.jpg', grad_norm)
    grad_s = 0.382683432 * grad_norm
    gs_c = snap(grad_c_img / (grad_s + 0.001))
    imageio.imwrite(img_filename + '.gsc.jpg', gs_c)
    gs_l = snap(grad_l_img / (grad_s + 0.001))
    imageio.imwrite(img_filename + '.gsl.jpg', gs_l)
    contours = np.zeros(image.shape, dtype=np.bool)
    for l in range(1, image.shape[0] - 1):
        for c in range(1, image.shape[1] - 1):
            contours[l, c] = (grad_norm[l, c] > seuil_bas
                              and grad_norm[l, c] > grad_norm[l + gs_l[l, c]][c + gs_c[l, c]]
                              and grad_norm[l, c] > grad_norm[l - gs_l[l, c]][c - gs_c[l, c]])
    if seuil_haut > seuil_bas:
        return grad_norm, Hyst(contours, grad_norm, seuil_haut)
    return grad_norm, contours


def ContoursLaplace(image, sigma, seuil_bas, seuil_haut):
    gauss_img = filters.gaussian(image, sigma=sigma, mode='nearest').astype(float)
    L = np.array([[1., 2., 1.], [2., -12., 2.], [1., 2., 1.]]) / 3.
    L_img = sp.ndimage.convolve(gauss_img, L)
    imageio.imwrite(img_filename + '.L.jpg', L_img)
    contours = np.zeros(image.shape, dtype=np.bool)
    pseudo_grad = seuil_bas * np.ones(image.shape)
    for l in range(1, image.shape[0] - 1):
        for c in range(1, image.shape[1] - 1):
            cur_L = L_img[l, c]
            if cur_L > 0:
                for dl in range(-1, 2):
                    for dc in range(-1, 2):
                        neigh_L = L_img[l + dl, c + dc]
                        pg = abs(gauss_img[l, c] - gauss_img[l + dl, c + dc])
                        if neigh_L < 0 and pg > pseudo_grad[l, c]:
                            contours[l, c] = 1
                            pseudo_grad[l, c] = pg
    if seuil_haut > seuil_bas:
        return Hyst(contours, grad_norm, seuil_haut)
    return contours


print('Reading %s' % img_filename)
image = imageio.imread(img_filename)
print(image.shape)
grad_norm, canny = ContoursCanny(image, sigma, seuil_bas, seuil_haut)
print('Writing %s' % canny_filename)
imageio.imwrite(canny_filename, sf.img_as_uint(canny))
laplace = ContoursLaplace(image, sigma, seuil_bas, seuil_haut)
print('Writing %s' % lapl_filename)
imageio.imwrite(lapl_filename, sf.img_as_uint(laplace))

# imageio.imwrite('raster_GreyClosing.tif', close_img)  # enregistre les résultats dans un fichier

plt.figure()
plt.subplot(221)
plt.imshow(image)
plt.title("image")
plt.subplot(222)
plt.imshow(grad_norm)
plt.title("grad_norm")
plt.subplot(223)
plt.imshow(canny)
plt.title("canny")
plt.subplot(224)
plt.imshow(laplace)
plt.title("laplace")
plt.show()
