import numpy as np


class Region():

    def __init__(self, region, data_mode=None, region_mode='square'):
        if data_mode == 'tlp':
            self.y0, self.x0, self.height, self.width = region
            self.height = self.height // 2 * 2 + 1
            self.width = self.width // 2 * 2 + 1
            self.x1 = self.x0 + self.width
            self.y1 = self.y0 + self.height
        else:
            self.x0, self.y0, self.width, self.height = region
            self.x1 = self.x0 + self.width
            self.y1 = self.y0 + self.height
        
        self.center_x = (self.x0 + self.x1) / 2
        self.center_y = (self.y0 + self.y1) / 2
        self.square_len = max(self.width, self.height)
        
        assert region_mode in ('square', 'raw')
        if region_mode == 'square':
            self.height = self.square_len
            self.width = self.square_len
            self.x0 = int(self.center_x - self.square_len / 2)
            self.x1 = int(self.center_x + self.square_len / 2)
            self.y0 = int(self.center_y - self.square_len / 2)
            self.y1 = int(self.center_y + self.square_len / 2)
        
    def __repr__(self):
        return ', '.join([str(self.x0), str(self.y0), str(self.x1), str(self.y1)])
        
    def add_box(self, image):
        line_width = min(image.shape[:2]) // 200 + 1
        image = image.copy()
        val = image.max()
        image[self.x0:self.x1+line_width, self.y0:self.y0+line_width, :] = (0, val, 0)
        image[self.x0:self.x1+line_width, self.y1:self.y1+line_width, :] = (0, val, 0)
        image[self.x0:self.x0+line_width, self.y0:self.y1+line_width, :] = (0, val, 0)
        image[self.x1:self.x1+line_width, self.y0:self.y1+line_width, :] = (0, val, 0)
        return image
    
    def get_label(self):
        return self.y0, self.x0, self.height, self.width


def get_loss(true_rect, pred_rect):
    if isinstance(pred_rect, Region):
        return np.sqrt((true_rect.center_x - pred_rect.center_x)**2 + (true_rect.center_y - pred_rect.center_y)**2)
    else:
        pred_x, pred_y = pred_rect
        return np.sqrt((true_rect.center_x - pred_x)**2 + (true_rect.center_y - pred_y)**2)