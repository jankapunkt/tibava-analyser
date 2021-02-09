# TIB-AV-A Backend

## TODOs

- [x] Face Detection with DLib
- [ ] Face Detection with *RetinaFace* ([Repo](https://github.com/deepinsight/insightface/tree/master/detection/RetinaFace))
- [ ] Shot Boundary Detection with *TransNet v2* ([Repo](https://github.com/soCzech/TransNetV2)) 
- [ ] Shot Type Classification

## Installation

To install and start the backend run:

```bash
docker-compose -f "docker-compose.yml" up -d --build
```

You can manually start the backend with:

```bash
python backend.py
```

## REST API

For reference you can check `test_backend.py`. Functions were tested using a local server with `<base_url> = http://128.0.0.1:5000/`

### Sanity Check

```
URL: <base_url>/ping
```

#### `GET`

Sanity check with: `<base_url>/ping` should return `pong!`

### Face Detection

```
URL: <base_url>/detect_faces/<int:video_id>
Parameters
----------
video_id : int
    The identifier of the video
```

#### `GET` (not implemented yet)

Get precalculated face detection results of a video with a specified `video_id` with: `<base_url>/detect_faces/<int:video_id>`

#### `PUT`

```
Parameters
----------
title : str
    The title of the video
path : str
    The path to the video file
max_frames : int, optional
    The maximum number of video frames to process (do not specify if the whole video should be processed)
```

Run and store face detection of a video with a `video_id`, `title`, and `path` with: `<base_url>/detect_faces/<int:video_id>` and `params={title: <str:title>, path: <str:path>}`.

Optionally you can limit the number of video frames to process using:
`params={title: <str:title>, path: <str:path>, max_frames: <int:max_frames>}`
