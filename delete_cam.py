import requests

camera_name = "Выезд весы (192.168.172.55)"
url = f"http://127.0.0.1:5000/delete_camera/{camera_name}"
response = requests.delete(url)

if response.status_code == 200:
    print(response.json()["message"])  # Камера удалена успешно
else:
    print(f"Ошибка: {response.status_code} - {response.json().get('error', 'Неизвестная ошибка')}")
