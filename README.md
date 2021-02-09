# TIB-AV-A Backend

## Installation

To install run:

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

Sanity check with: `<base_url>/ping` should return `pong!`

### Face Detection

`GET`: Get precalculated face detection results of a video with a specified `video_id` with: `<base_url>/detect_faces/video_id`

`PUT`: Run and store face detection of a video with a `video_id`, `title`, and `path` using a `PUT` request with: `<base_url>/detect_faces/<int:video_id>` and `params={title: <str:title>, path: <str:path>}`.

Optionally you can limit the number of video frames to process using:
`params={title: <str:title>, path: <str:path>, max_frames: <int:max_frames>}`
