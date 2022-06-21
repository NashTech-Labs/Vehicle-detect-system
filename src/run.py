from model.classifier import classifier
from utils.constants import filename, cascades

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    obj = classifier()
    count = obj.detect_model(filename, cascades) #Detect model is the function which will detect the total number of vehicles in the image
    print("No. of Cars Found in the Image :", count)
