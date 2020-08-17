import glob
import requests
import json


def predict_captcha() -> str:
    url = "http://127.0.0.1:9000/predict"
    filename = "/img_captcha/test-data/tmp.jpg"
    files = {'image': (filename, open(filename, 'rb'))}
    response = requests.post(url, files=files)
    res_json = json.loads(response.text)
    prediction = res_json["prediction"]
    print(prediction)


predict_captcha()
