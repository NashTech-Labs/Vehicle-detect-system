# Import libraries
from data.load_data import load_data
from processing_data.pre_processing import pre_processing
import cv2
from PIL import Image

class classifier:
    def __init__(self):
        pass

    def detect_model(self, filename, cascades):
        load_obj = load_data()
        processing_obj = pre_processing()

        image, image_arr = load_obj.get_image(filename)  # reading the Image here
        closing = processing_obj.processing(filename)  # Pre-Processing the Image here
        car_cascade = cv2.CascadeClassifier(cascades)  # Training of the images from the pretrained xml file
        cars = car_cascade.detectMultiScale(closing, 1.1, 1)  # To detect multiple objects
        cnt = 0
        for (x, y, w, h) in cars:
            cv2.rectangle(image_arr, (x, y), (x + w, y + h), (255, 0, 0),
                          2)  # cv2.rectangle(image, start_point, end_point, color, thickness)
            cnt += 1
        image = Image.fromarray(image_arr)
        image.save("Detect_Count_Image.jpg")
        return cnt