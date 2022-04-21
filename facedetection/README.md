# Note
- Currently the CPU-version of mxnet (no CUDA) is used to calculate the bboxes of the faces
  - To use the GPU-version replace `mxnet==1.8.0.post0` with `mxnet-cu112==1.8.0.post0` (for CUDA-11.2) inside the `requirements.txt`
  - Also install the specific pytorch version 1.9.0 which supports CUDA 
  - set gpuid >= 0 inside `facedetection_client.py` and `facedetection_server.py` to use GPU

# Setup
- Download the model for retinaface from [onedrive](https://1drv.ms/u/s!AswpsDO2toNKrUy0VktHTWgIQ0bn?e=UEF7C4)
  - Take the files from the folder `retinaface_r50_v1` and put them into `./model/`
- Download one model for arcface from [onedrive](https://onedrive.live.com/?authkey=%21AFZjr283nwZHqbA&id=4A83B6B633B029CC%215577&cid=4A83B6B633B029CC) (`ms1mv3_arcface_r100_fp16` was used during implementation)
  - Take the file `backbone.pth` and put it into `./model/`
- Create docker-image with the Dockerfile
