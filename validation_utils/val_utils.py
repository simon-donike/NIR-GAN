import numpy as np
"""
def crop_center( im, target_height):
        target_width = target_height
        assert len(im.shape) == 3, "Input must be a 3D array, actuall shape: {}".format(im.shape)
        c, h, w = im.shape  # Get the dimensions of the image
        assert c < h and c < w, "Input must be a 3D array with the first dimension being the number of channels"
        assert target_height <= h and target_width <= w, "Target dimensions must be smaller than image dimensions"
        
        # Calculate the starting points for the crop
        start_h = (h - target_height) // 2
        start_w = (w - target_width) // 2
        
        # Perform the crop
        cropped_im = im[:, start_h:start_h + target_height, start_w:start_w + target_width]
        return cropped_im
"""


def crop_center(im, target_height):
    import numpy as np
    target_width = target_height

    # Track if input was 2D
    was_2d = False
    if len(im.shape) == 2:
        im = np.expand_dims(im, axis=0)
        was_2d = True

    assert len(im.shape) == 3, f"Expected 3D array, got shape {im.shape}"
    c, h, w = im.shape
    assert target_height <= h and target_width <= w, "Target size must be <= image size"

    start_h = (h - target_height) // 2
    start_w = (w - target_width) // 2

    cropped = im[:, start_h:start_h + target_height, start_w:start_w + target_width]

    if was_2d:
        return cropped[0]  # return as H×W
    else:
        return cropped      # return as C×H×W