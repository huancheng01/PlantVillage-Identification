import requests

resp = requests.post("http://192.168.1.110:6666/predict",
                     files={"file": open('APAS_image (5)_16_0.jpg','rb')})

print(resp.json())