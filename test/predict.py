import requests
import time
import pprint

# Initialize the keras REST API endpoint URL.
REST_API_URL = 'http://127.0.0.1:5000/ship_detect/predict'


def predict_result(image_path):
    # Initialize image path
    image = open(image_path, 'rb').read()
    payload = {'image': image}

    # Submit the request.
    try:
        r = requests.post(REST_API_URL, files=payload).json()

        pprint.pprint(r)
    except:
        pass


if __name__ == '__main__':

    t1 = time.time()
    img_path = r'E:\YOLO\test_image\timg1.jpg'
    predict_result(img_path)
    t2 = time.time()
    print(t2-t1)