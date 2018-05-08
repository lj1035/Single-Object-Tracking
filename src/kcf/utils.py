import cv2
import numpy as np
from skimage.feature import hog

NORM_TYPE = 'ortho'


def dense_gauss_kernel(sigma, xf, x, zf=None, z=None):
    
    N = xf.shape[0] * xf.shape[1]
    xx = np.dot(x.flatten().transpose(), x.flatten())  # squared norm of x

    if zf is None:
        # auto-correlation of x
        zf = xf
        zz = xx
    else:
        zz = np.dot(z.flatten().transpose(), z.flatten())  # squared norm of y

    xyf = np.multiply(zf, np.conj(xf))
    if len(xyf.shape) == 3:
        xyf_ifft = np.fft.ifft2(np.sum(xyf, axis=2), norm=NORM_TYPE)
    if len(xyf.shape) == 2:
        xyf_ifft = np.fft.ifft2(xyf, norm=NORM_TYPE)

    c = np.real(xyf_ifft)
    d = np.real(xx) + np.real(zz) - 2 * c
    k = np.exp(-1. / sigma ** 2 * np.abs(d) / N)

    return k


def get_subwindow(image, pos, size, cell_size, feature='raw'):
    if np.isscalar(size):
        size = [size, size]
    
    xs = np.floor(pos[1]) + np.arange(size[1], dtype=int) - np.floor(size[1] / 2)
    ys = np.floor(pos[0]) + np.arange(size[0], dtype=int) - np.floor(size[0] / 2)
    xs = xs.astype(int)
    ys = ys.astype(int)
    
    # check for out-of-bounds coordinates and set them to the values at the borders
    xs[xs < 0] = 0
    ys[ys < 0] = 0
    xs[xs >= image.shape[1]] = image.shape[1] - 1
    ys[ys >= image.shape[0]] = image.shape[0] - 1

    
    out = image[np.ix_(ys, xs)]
    
    if feature == 'hog':
        # d0 = hog(out[:, :, 0], pixels_per_cell=(cell_size, cell_size), cells_per_block=(1, 1), transform_sqrt=True, 
        #          visualise=False, block_norm='L2-Hys', feature_vector=False).squeeze()
        # d1 = hog(out[:, :, 1], pixels_per_cell=(cell_size, cell_size), cells_per_block=(1, 1), transform_sqrt=True, 
        #          visualise=False, block_norm='L2-Hys', feature_vector=False).squeeze()
        # d2 = hog(out[:, :, 2], pixels_per_cell=(cell_size, cell_size), cells_per_block=(1, 1), transform_sqrt=True, 
        #          visualise=False, block_norm='L2-Hys', feature_vector=False).squeeze()
        # #print(d0.shape)
        # out = np.mean((d0, d1, d2), axis=0)

        #print(out.shape)
        out = cv2.resize(out, (int(np.floor(out.shape[1] / cell_size)),
                               int(np.floor(out.shape[0] / cell_size))))
        d0 = hog(out[:, :, 0], pixels_per_cell=(cell_size, cell_size), transform_sqrt=True, 
                 visualise=True, block_norm='L2-Hys', feature_vector=False)[1]
        d1 = hog(out[:, :, 1], pixels_per_cell=(cell_size, cell_size), transform_sqrt=True, 
                 visualise=True, block_norm='L2-Hys', feature_vector=False)[1]
        d2 = hog(out[:, :, 2], pixels_per_cell=(cell_size, cell_size), transform_sqrt=True, 
                 visualise=True, block_norm='L2-Hys', feature_vector=False)[1]
        out = np.stack((d0, d1, d2), axis=2)

    
    if feature == 'gray':
        out -= out.mean()
        
    return out


def create_labels(spatial_bandwidth, window_size, target_size, cell_size):

    # Compute output sigma based on spatial bandwidth
    output_sigma = np.sqrt(np.prod(target_size)) * spatial_bandwidth / cell_size
    
    # Generate regression targets
    output_size = np.floor(window_size / cell_size)
    grid_x = np.arange(np.floor(output_size[1])) - np.floor(output_size[1] / 2)
    grid_y = np.arange(np.floor(output_size[0])) - np.floor(output_size[0] / 2)
    rs, cs = np.meshgrid(grid_x, grid_y)
    labels = np.exp(-0.5 / output_sigma ** 2 * (rs ** 2 + cs ** 2))
    
    # Move the peak to the top-left corner
    
#     shift_amount = -np.floor(output_size[1] / 2) + 1
#     shift_amount = shift_amount.astype(int)
#     labels = np.roll(labels, shift_amount)
#     labels = np.roll(labels, labels.shape[0]//2, 0)
#     labels = np.roll(labels, labels.shape[1]//2, 1)

    return labels
