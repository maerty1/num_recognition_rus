import requests

camera_name = "Въезд весы (192.168.178.148)"
url = f"http://127.0.0.1:5000/delete_camera/{camera_name}"
response = requests.delete(url)

if response.status_code == 200:
    print(response.json()["message"])  # Камера удалена успешно
else:
    print(f"Ошибка: {response.status_code} - {response.json().get('error', 'Неизвестная ошибка')}")
