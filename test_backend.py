import requests

base_url = "http://127.0.0.1:5000/"

response = requests.get(base_url + "ping")
print(response.json())

response = requests.get(base_url + "detect_faces/1")
response = requests.put(
    base_url + "detect_faces/1",
    {
        "title": "title",
        "path": "media/Crash_Course_Engineering_Preview_-_English.mp4",
        "max_frames": 100  # omit param for whole video
    })

print(response)
print(response.json())
