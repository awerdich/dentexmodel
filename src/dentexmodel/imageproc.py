"""
Methods for image processing
Andreas Werdich
Center for Computational Biomedicine
"""

import os
import copy
import numpy as np
import cv2
import logging
from skimage import io
from PIL import Image

logger = logging.getLogger(name=__name__)


def clip_range(r, max_val):
    return max(min(r, max_val), 0)


def transform_box(box_padded, pad_pixels, img):
    """
    Transform bounding box from padded to original image
    """
    img_h, img_w = img.shape[:2]
    x, y, w, h = box_padded - float(pad_pixels)
    output_box = tuple([clip_range(x, img_w),
                        clip_range(y, img_h),
                        clip_range(w, img_w),
                        clip_range(h, img_h)])
    return output_box


def crop_image(image, box):
    """
    Crops an object in an image by bounding box
    :Parameters:
        image: (np.ndarray) image data
        box: (tuple) (x_min, y_min, width, height)
    :returns:
        crop_img: (np.ndarray) cropped image
    """
    x, y, w, h = [int(np.round(c)) for c in box]
    return image[y:h, x:w, :]


def is_image(image_file_path):
    """
    Checks if the given file path corresponds to an image file.
    Parameters:
        image_file_path (str): The path to the file.
    Returns:
        bool: True if the file is an image, False otherwise.
    """
    file_is_image = False
    if os.path.exists(image_file_path):
        try:
            Image.open(image_file_path)
        except Exception as ex:
            logger.warning(f'File: {image_file_path} is not an image.')
        else:
            file_is_image = True
    else:
        logger.warning(f'File: {image_file_path} does not exist.')
    return file_is_image


def validate_image_data(data_df, file_path_col):
    """ Load and validate images from data frame
    :parameters:
        data_df (pd.DataFrame): data frame with image file paths
        file_path_col (list): List of columns with file paths
    :returns
        output_df (pd.DataFrame): data frame with valid file paths
    """
    output_df = copy.deepcopy(data_df)
    file_path_col = [file_path_col] if isinstance(file_path_col, str) else file_path_col
    for col in file_path_col:
        n_start = len(output_df)
        output_df = output_df.loc[output_df[col].apply(is_image)]
        n_dropped = n_start - len(output_df)
        if n_dropped > 0:
            warning_msg = f'Dropped {n_dropped} rows from bad data in column: {col}.'
            logger.warning(warning_msg)
        else:
            logger.info('All files validated.')
    return output_df


class ImageData:
    """ Load and transform images """

    def __init__(self, resize=None):
        self.resize = resize
        self.image_net_mean = [0.485, 0.456, 0.406]
        self.image_net_std = [0.229, 0.224, 0.225]

    def load_image(self, image_path):
        """ Load image as np.ndarray
        Parameters:
            image_path: (str) complete path to image file
        Returns:
            output_array: (np.ndarray) (uint8)
        """
        output_array = None
        if os.path.exists(image_path):
            try:
                with open(image_path, mode='rb') as fl:
                    img = io.imread(fl)
            except Exception as ex:
                read_error_msg = f'Unable to read: {image_path}'
                print(read_error_msg)
                logger.error(read_error_msg)
            else:
                output_array = img.astype(np.uint8)
        else:
            error_msg = f'Image file: {image_path} does not exist.'
            print(error_msg)
            logger.error(error_msg)
        return output_array

    def image_size(self, image_path):
        """ Load image and return size in pixels (height, width) """
        output = None
        if is_image(image_path):
            output_array = self.load_image(image_path)
            if output_array is not None:
                output = tuple(output_array.shape)
        return output

    def np_square_pad(self, im_array, pad_pixels, pad_number=0):
        """ Pad 2D image
        Parameters:
            im_array: (np.ndarray) 2D numpy array
            pad_pixels: (int) pixels to add on each side
            pad_number: (int) gray value in [0, 256]
        """
        assert len(im_array.shape) == 2, f'Require 2D grayscale image.'

        def pad_with(vector, pad_width, iaxis, kwargs):
            pad_value = kwargs.get('padder', 10)
            vector[:pad_width[0]] = pad_value
            vector[-pad_width[1]:] = pad_value

        output = np.pad(im_array, pad_pixels, pad_with, padder=pad_number)
        return output

    def np2color(self, im_array, color_scheme='RGB'):
        """ Convert np.ndarray into color image
        Parameters:
            im_array: (np.ndarray) 2D or 3D numpy array
            color_scheme: (str) 'RGB', 'BGR' or 'GRAY'
        """
        assert color_scheme in ['RGB', 'BGR', 'GRAY'], f'color_scheme must be RGB, BGR or GRAY'
        if len(im_array.shape) == 2:
            color_code_str = f'cv2.COLOR_GRAY2{color_scheme}'
            image = cv2.cvtColor(im_array, code=eval(color_code_str))
        elif len(im_array.shape) == 3:
            if color_scheme == 'RGB':
                image = im_array.copy()
            else:
                color_code_str = f'cv2.COLOR_RGB2{color_scheme}'
                image = cv2.cvtColor(im_array, code=eval(color_code_str))
        else:
            raise NotImplementedError('input array must be 2D or 3D.')
        return image

    def convert_transparent_png(self, img):
        """ Decomposing the alpha channel for 4 channel png file """
        # image_4channel = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
        output_image = img
        if len(img.shape) == 3 and img.shape[2] == 4:
            alpha_channel = img[:, :, 3]
            rgb_channels = img[:, :, :3]
            white_background_image = np.ones_like(rgb_channels, dtype=np.uint8) * 255
            alpha_factor = alpha_channel[:, :, np.newaxis].astype(np.float32) / 255.0
            alpha_factor = np.concatenate((alpha_factor, alpha_factor, alpha_factor), axis=2)
            base = rgb_channels.astype(np.float32) * alpha_factor
            white = white_background_image.astype(np.float32) * (1 - alpha_factor)
            output_image = (base + white).astype(np.uint8)
        return output_image

    def resize_image_list(self, image_list, output_size):
        """ Match size for multiple inputs before augmentation
        Parameters:
            image_list, list of images
            output_size, int or tuple, size of output image
        """
        assert any([isinstance(output_size, int),
                    isinstance(output_size, tuple)]), 'output_size must be of type int or tuple.'

        if isinstance(output_size, int):
            dim = (output_size, output_size)
        else:
            dim = output_size

        output_image_list = [cv2.resize(im, dim, interpolation=cv2.INTER_AREA) for im in image_list]
        return output_image_list

    def hist_eq(self, img):
        """ Adaptive histogram equalization
        Parameters:
            img (np.ndarray) RGB image
        Returns:
            enhanced_img (np.ndarray) RGB image with enhanced contrast
        """
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        l_channel, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l_channel)
        lim_g = cv2.merge((cl, a, b))
        enhanced_img = cv2.cvtColor(lim_g, cv2.COLOR_LAB2RGB)
        return enhanced_img

