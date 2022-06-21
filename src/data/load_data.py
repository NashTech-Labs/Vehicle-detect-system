# Import libraries
from PIL import Image
import numpy as np

class load_data:
    def __init__(self):
        pass

    def get_image(self, filename):
        # Reading image
        image = Image.open(filename)
        image = image.resize((450, 250))
        image_arr = np.array(image)
        return image, image_arr
