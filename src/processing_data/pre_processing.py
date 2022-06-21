# Import libraries
from PIL import Image
import cv2
import numpy as np
from data.load_data import load_data


class pre_processing:
    def __init__(self):
        pass

    def processing(self, filename):
        obj = load_data()
        image, image_arr = obj.get_image(filename)

        # converting image into grayscale
        grey = cv2.cvtColor(image_arr, cv2.COLOR_BGR2GRAY)
        Image.fromarray(grey)

        # applying GaussianBlur to remove the noise from the image
        blur = cv2.GaussianBlur(grey, (5, 5), 0)
        Image.fromarray(blur)

        # dilate image to fill the missing parts of the images
        dilated = cv2.dilate(blur, np.ones((3, 3)))
        Image.fromarray(dilated)

        # Appling closing here
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        closing = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel)
        Image.fromarray(closing)

        return closing
