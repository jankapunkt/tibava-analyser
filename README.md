# TIB-AV-A Backend

## Installation

To install run:

```bash
docker-compose -f "docker-compose.yml" up -d --build
```

Start backend with:

```bash
python backend.py
```

## REST API

For reference you can check `test.py`. Functions were tested using a local server with `<base_url> = http://128.0.0.1:5000/`

### Sanity Check

Sanity check with: `<base_url>/ping` should return `pong`

### Face Detection

`GET`: Get precalculated face detection results of a video with a specified `video_id` with: `<base_url>/detect_faces/video_id`

`POST`: Run and store face detection of a video with a `title` and `path` using a post request with: `<base_url>/detect_faces/video_id` and `params={title: <title>, path: <path>}`
