import numpy as np

def crop_center( im, target_height):
        target_width = target_height
        assert len(im.shape) == 3, "Input must be a 3D array"
        c, h, w = im.shape  # Get the dimensions of the image
        assert c < h and c < w, "Input must be a 3D array with the first dimension being the number of channels"
        assert target_height <= h and target_width <= w, "Target dimensions must be smaller than image dimensions"
        
        # Calculate the starting points for the crop
        start_h = (h - target_height) // 2
        start_w = (w - target_width) // 2
        
        # Perform the crop
        cropped_im = im[:, start_h:start_h + target_height, start_w:start_w + target_width]
        return cropped_im