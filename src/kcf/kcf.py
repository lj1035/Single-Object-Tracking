import math
import os
import sys
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from skimage.feature import hog

from ..utils import *
from .utils import *

NORM_TYPE = 'ortho'


class KCFtracker:
    
    def __init__(self, image, region, cell_size, padding, feature='raw'):
        
        self.feature = feature
        if self.feature == 'raw':
            self.cell_size = 1
            self.feature_bandwidth = 0.2
            self.adaptation_rate = 0.075
        else:
            self.cell_size = cell_size
            self.feature_bandwidth = 0.5
            self.adaptation_rate = 0.02
        self.target_size = np.array([region.width, region.height])
        self.pos = [region.center_x, region.center_y]
        self.window_size = np.floor(self.target_size * (1 + padding)) #// 2 * 2 + 1

        img_crop = get_subwindow(image, self.pos, self.window_size, self.cell_size, self.feature)
        
        spatial_bandwidth = math.sqrt(region.width * region.height) / 10.
        
        # Create lables following Gaussian distribution
        self.y = create_labels(spatial_bandwidth, self.window_size, self.target_size, self.cell_size)
        
        # Compute cosine window to mitigage discontinuities at the image boundaries
        self.cos_window = np.outer(np.hanning(self.y.shape[0]), np.hanning(self.y.shape[1]))
        
        # Get training image patch x
        self.x = np.multiply(img_crop, self.cos_window[:, :, None])

        # FFT Transformation
        # First transform y
        self.yf = np.fft.fft2(self.y, axes=(0, 1), norm=NORM_TYPE)
        # Then transfrom x
        self.xf = np.fft.fft2(self.x, axes=(0, 1), norm=NORM_TYPE)
    
        kf = np.fft.fft2(dense_gauss_kernel(self.feature_bandwidth, self.xf, self.x), axes=(0, 1), norm=NORM_TYPE)

        self.lambda_value = 1e-4
        self.alphaf = np.divide(self.yf, kf + self.lambda_value)
        self.model_alphaf = self.alphaf
        self.model_xf = self.xf

    def track(self, image):
#         plt.imshow(np.real(np.fft.ifft2(self.model_alphaf)))
#         plt.show()
        
        # Interpolation
        test_crop = get_subwindow(image, self.pos, self.window_size, self.cell_size, self.feature)
        z = np.multiply(test_crop, self.cos_window[:, :, None])
        zf = np.fft.fft2(z, axes=(0, 1), norm=NORM_TYPE)
        k_test = dense_gauss_kernel(self.feature_bandwidth, self.model_xf, self.x, zf, z)
        kf_test = np.fft.fft2(k_test, axes=(0, 1), norm=NORM_TYPE)
        
        response = np.real(np.fft.ifft2(np.multiply(self.model_alphaf, kf_test), norm=NORM_TYPE))
        
        alphaf =  np.divide(self.yf, kf_test + self.lambda_value)
        self.model_alphaf = (1 - self.adaptation_rate) * self.model_alphaf + self.adaptation_rate * alphaf
        self.model_xf = (1 - self.adaptation_rate) * self.model_xf + self.adaptation_rate * zf
 
#         plt.imshow(k_test)
#         plt.show()
#         plt.imshow(response)
#         plt.show()

        # Max position in response map
        v_center, h_center = np.unravel_index(response.argmax(), response.shape)
        vert_delta, horiz_delta = [v_center - response.shape[0] / 2, h_center - response.shape[1] / 2]
        
        # Switch the quadrants of the output when the object does not move
        #print(vert_delta, horiz_delta)
        if vert_delta > zf.shape[0] / 2:
            vert_delta -= zf.shape[0]
        if horiz_delta > zf.shape[1] / 2:
            horiz_delta -= zf.shape[1]
        #print(vert_delta, horiz_delta)
        # Predicted position
        self.pos = [self.pos[0] + vert_delta, self.pos[1] + horiz_delta]
        
        return Region((int(self.pos[1] - self.target_size[1] / 2),
                       int(self.pos[0] - self.target_size[0] / 2), 
                       int(self.target_size[1]),
                       int(self.target_size[0])), data_mode='tlp', region_mode='square')
