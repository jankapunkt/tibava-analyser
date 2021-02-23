import requests

backend_url = "http://127.0.0.1:5000/"
shot_detection_url = "http://127.0.0.1:5001/"
face_detection_url = "http://127.0.0.1:5002/"

print("Ping backend")
response = requests.get(backend_url + "ping")
print(response.json())

print("Ping shot detection")
response = requests.get(shot_detection_url + "ping")
print(response.json())

print("Ping face detection")
response = requests.get(face_detection_url + "ping")
print(response.json())

print("Test shot detection")
response = requests.put(backend_url + "detect_shots/1", {
    "title": "Crash_Course_Engineering_Preview_-_English",
    "path": "media/Crash_Course_Engineering_Preview_-_English.mp4"
})
print(response)
print(response.json())

print("Test face detection")
response = requests.put(
    backend_url + "detect_faces/1", {
        "title": "Crash_Course_Engineering_Preview_-_English",
        "path": "media/Crash_Course_Engineering_Preview_-_English.mp4",
        "max_frames": 100
    })

print(response)
print(response.json())
