import requests
import json

# URL сервера
server_url = "http://127.0.0.1:5000/add_camera"

# Данные камер
cameras = [
    {
        "url": "http://192.168.178.148/action/snap?cam=0&user=admin&pwd=admin",
        "x0": 5,
        "y0": 300,
        "x1": 2500,
        "y1": 1800,
        "name": "Въезд весы (192.168.178.148)"
    },
    {
        "url": "http://192.168.178.147/action/snap?cam=0&user=admin&pwd=admin",
        "x0": 500,
        "y0": 100,
        "x1": 2590,
        "y1": 1500,
        "name": "Выезд весы (192.168.178.147)"
    },
    {
        "url": "http://192.168.178.149/action/snap?cam=0&user=admin&pwd=admin",
        "x0": 600,
        "y0": 300,
        "x1": 1650,
        "y1": 1050,
        "name": "Въезд дисп (192.168.178.149)"
    }
]

# Отправка запросов для добавления камер
for camera in cameras:
    headers = {'Content-Type': 'application/json'}
    response = requests.post(server_url, headers=headers, data=json.dumps(camera))
    if response.status_code == 201:
        print(f"Camera {camera['name']} added successfully.")
    elif response.status_code == 409:
        print(f"Camera {camera['name']} already exists.")
    else:
        print(f"Failed to add camera {camera['name']}: {response.json()}")

