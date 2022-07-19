from analyser.plugins.manager import AnalyserPluginManager
from analyser.utils import VideoDecoder
from analyser.data import BboxData, BboxesData, ImageData, ImagesData, VideoData, generate_id, create_data_path
from analyser.plugins import Plugin
from analyser.utils import VideoDecoder
import cv2
import imageio
import logging
import numpy as np
import sys
import traceback
from analyser.utils import InferenceServer, Backend, Device


default_config = {
    "data_dir": "/data/",
    "host": "localhost",
    "port": 6379,
    "model_name": "insightface",
    "model_device": "cpu",
    "model_file": "/models/insightface_detector/scrfd_10g_bnkps.onnx",
}

default_parameters = {"fps": 1.0, "det_thresh": 0.5, "nms_thresh": 0.4, "input_size": (640, 640)}

requires = {
    "video": VideoData,
}

provides = {
    "images": ImagesData,
    "bboxes": BboxesData,
}


@AnalyserPluginManager.export("insightface_detector")
class InsightfaceDetector(
    Plugin, config=default_config, parameters=default_parameters, version="0.1", requires=requires, provides=provides
):
    def __init__(self, config=None):
        super().__init__(config)
        self.host = self.config["host"]
        self.port = self.config["port"]
        self.model_name = self.config["model_name"]
        self.model_device = self.config["model_device"]
        self.model_file = self.config["model_file"]

        self.server = InferenceServer(
            model_file=self.model_file, model_name=self.model_name, host=self.host, port=self.port, backend=Backend.ONNX
        )

    def distance2bbox(self, points, distance, max_shape=None):
        """Decode distance prediction to bounding box.
        Args:
            points (Tensor): Shape (n, 2), [x, y].
            distance (Tensor): Distance from the given point to 4
                boundaries (left, top, right, bottom).
            max_shape (tuple): Shape of the image.
        Returns:
            Tensor: Decoded bboxes.
        """
        x1 = points[:, 0] - distance[:, 0]
        y1 = points[:, 1] - distance[:, 1]
        x2 = points[:, 0] + distance[:, 2]
        y2 = points[:, 1] + distance[:, 3]
        if max_shape is not None:
            x1 = x1.clamp(min=0, max=max_shape[1])
            y1 = y1.clamp(min=0, max=max_shape[0])
            x2 = x2.clamp(min=0, max=max_shape[1])
            y2 = y2.clamp(min=0, max=max_shape[0])
        return np.stack([x1, y1, x2, y2], axis=-1)

    def distance2kps(self, points, distance, max_shape=None):
        """Decode distance prediction to bounding box.
        Args:
            points (Tensor): Shape (n, 2), [x, y].
            distance (Tensor): Distance from the given point to 4
                boundaries (left, top, right, bottom).
            max_shape (tuple): Shape of the image.
        Returns:
            Tensor: Decoded bboxes.
        """
        preds = []
        for i in range(0, distance.shape[1], 2):
            px = points[:, i % 2] + distance[:, i]
            py = points[:, i % 2 + 1] + distance[:, i + 1]
            if max_shape is not None:
                px = px.clamp(min=0, max=max_shape[1])
                py = py.clamp(min=0, max=max_shape[0])
            preds.append(px)
            preds.append(py)
        return np.stack(preds, axis=-1)

    def forward(self, img, threshold, use_kps=True):
        output_names = ["448", "471", "494", "451", "474", "497", "454", "477", "500"]
        scores_list = []
        bboxes_list = []
        kpss_list = []
        center_cache = {}
        input_mean = 127.5
        input_std = 128.0
        fmc = 3
        feat_stride_fpn = [8, 16, 32]
        num_anchors = 2
        batched = False
        input_size = tuple(img.shape[0:2][::-1])
        blob = cv2.dnn.blobFromImage(
            img, 1.0 / input_std, input_size, (input_mean, input_mean, input_mean), swapRB=True
        )
        # print(blob, blob.shape)
        # self.con.tensorset(f"data_{job_id}", blob)
        result = self.server({"data": blob}, output_names)
        # result = self.con.modelrun(self.model_name, f"data_{job_id}", output_names)
        # net_outs = self.session.run(self.output_names, {self.input_name : blob})  # original function
        net_outs = [result.get(output_name) for output_name in output_names]

        input_height = blob.shape[2]
        input_width = blob.shape[3]
        for idx, stride in enumerate(feat_stride_fpn):
            # If model support batch dim, take first output
            if batched:
                scores = net_outs[idx][0]
                bbox_preds = net_outs[idx + fmc][0]
                bbox_preds = bbox_preds * stride
                if use_kps:
                    kps_preds = net_outs[idx + fmc * 2][0] * stride
            # If model doesn't support batching take output as is
            else:
                scores = net_outs[idx]
                bbox_preds = net_outs[idx + fmc]
                bbox_preds = bbox_preds * stride
                if use_kps:
                    kps_preds = net_outs[idx + fmc * 2] * stride

            height = input_height // stride
            width = input_width // stride
            K = height * width
            key = (height, width, stride)
            if key in center_cache:
                anchor_centers = center_cache[key]
            else:
                anchor_centers = np.stack(np.mgrid[:height, :width][::-1], axis=-1).astype(np.float32)
                anchor_centers = (anchor_centers * stride).reshape((-1, 2))
                if num_anchors > 1:
                    anchor_centers = np.stack([anchor_centers] * num_anchors, axis=1).reshape((-1, 2))
                if len(center_cache) < 100:
                    center_cache[key] = anchor_centers

            pos_inds = np.where(scores >= threshold)[0]
            bboxes = self.distance2bbox(anchor_centers, bbox_preds)
            pos_scores = scores[pos_inds]
            pos_bboxes = bboxes[pos_inds]
            scores_list.append(pos_scores)
            bboxes_list.append(pos_bboxes)
            if use_kps:
                kpss = self.distance2kps(anchor_centers, kps_preds)
                # kpss = kps_preds
                kpss = kpss.reshape((kpss.shape[0], -1, 2))
                pos_kpss = kpss[pos_inds]
                kpss_list.append(pos_kpss)
        return scores_list, bboxes_list, kpss_list

    def nms(self, dets, nms_thresh):
        thresh = nms_thresh
        x1 = dets[:, 0]
        y1 = dets[:, 1]
        x2 = dets[:, 2]
        y2 = dets[:, 3]
        scores = dets[:, 4]

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)

            inds = np.where(ovr <= thresh)[0]
            order = order[inds + 1]

        return keep

    def detect(
        self,
        frame,
        image_id,
        input_size=(640, 640),
        max_num=0,
        metric="default",
        det_thresh=0.5,
        nms_thresh=0.4,
        fps=10,
    ):
        img = frame.get("frame")
        im_ratio = float(img.shape[0]) / img.shape[1]
        model_ratio = float(input_size[1]) / input_size[0]
        if im_ratio > model_ratio:
            new_height = input_size[1]
            new_width = int(new_height / im_ratio)
        else:
            new_width = input_size[0]
            new_height = int(new_width * im_ratio)
        det_scale = float(new_height) / img.shape[0]
        resized_img = cv2.resize(img, (new_width, new_height))
        det_img = np.zeros((input_size[1], input_size[0], 3), dtype=np.uint8)
        det_img[:new_height, :new_width, :] = resized_img
        use_kps = True
        scores_list, bboxes_list, kpss_list = self.forward(det_img, det_thresh, use_kps)

        scores = np.vstack(scores_list)
        scores_ravel = scores.ravel()
        order = scores_ravel.argsort()[::-1]
        bboxes = np.vstack(bboxes_list) / det_scale
        if use_kps:
            kpss = np.vstack(kpss_list) / det_scale
        pre_det = np.hstack((bboxes, scores)).astype(np.float32, copy=False)
        pre_det = pre_det[order, :]
        keep = self.nms(pre_det, nms_thresh)
        det = pre_det[keep, :]
        if use_kps:
            kpss = kpss[order, :, :]
            kpss = kpss[keep, :, :]
        else:
            kpss = None
        if max_num > 0 and det.shape[0] > max_num:
            area = (det[:, 2] - det[:, 0]) * (det[:, 3] - det[:, 1])
            img_center = img.shape[0] // 2, img.shape[1] // 2
            offsets = np.vstack(
                [(det[:, 0] + det[:, 2]) / 2 - img_center[1], (det[:, 1] + det[:, 3]) / 2 - img_center[0]]
            )
            offset_dist_squared = np.sum(np.power(offsets, 2.0), 0)
            if metric == "max":
                values = area
            else:
                values = area - offset_dist_squared * 2.0  # some extra weight on the centering
            bindex = np.argsort(values)[::-1]  # some extra weight on the centering
            bindex = bindex[0:max_num]
            det = det[bindex, :]
            if kpss is not None:
                kpss = kpss[bindex, :]
        # create bbox objects (added to original code)
        predictions = []
        for bbox in det:
            x, y = int(max(0, bbox[0])), int(max(0, bbox[1]))
            w, h = int(bbox[2] - x), int(bbox[3] - y)
            # det_score = bbox[4]  # TODO: check if it is necessary to get the score
            predictions.append(
                BboxData(
                    image_id=image_id,
                    time=frame.get("time"),
                    delta_time=1 / fps,
                    x=x / img.shape[1],
                    y=y / img.shape[0],
                    w=w / img.shape[1],
                    h=h / img.shape[0],
                )
            )
        return predictions

    def call(self, inputs, parameters):
        try:
            images = []
            bboxes = []
            # decode video to extract bboxes per frame
            video_decoder = VideoDecoder(path=inputs["video"].path, fps=parameters.get("fps"))
            # iterate through frames to get images and bboxes
            for frame in video_decoder:
                image_id = generate_id()
                frame_bboxes = self.detect(
                    frame,
                    image_id,
                    parameters.get("input_size"),
                    det_thresh=parameters.get("det_thresh"),
                    nms_thresh=parameters.get("nms_thresh"),
                )

                for bbox in frame_bboxes:
                    # store image and bboxes
                    bbox_id = generate_id()
                    output_path = create_data_path(self.config.get("data_dir"), bbox_id, "jpg")
                    frame_image = frame.get("frame")
                    h, w = frame_image.shape[:2]
                    imageio.imwrite(
                        output_path,
                        frame_image[
                            int(bbox.y * h + 0.5) : int((bbox.y + bbox.h) * h + 0.5),
                            int(bbox.x * w + 0.5) : int((bbox.x + bbox.w) * w + 0.5),
                            :,
                        ],
                    )
                    images.append(
                        ImageData(id=bbox_id, ext="jpg", time=frame.get("time"), delta_time=1 / parameters.get("fps"))
                    )

                # get bboxes
                bboxes.extend(
                    self.detect(
                        frame,
                        image_id,
                        parameters.get("input_size"),
                        det_thresh=parameters.get("det_thresh"),
                        nms_thresh=parameters.get("nms_thresh"),
                        fps=parameters.get("fps"),
                    )
                )

            images_data = ImagesData(images=images)
            bboxes_data = BboxesData(bboxes=bboxes)
            return {"images": images_data, "bboxes": bboxes_data}
        except Exception as e:
            logging.error(f"Indexer: {repr(e)}")
            exc_type, exc_value, exc_traceback = sys.exc_info()

            traceback.print_exception(
                exc_type, exc_value, exc_traceback, limit=2, file=sys.stdout,
            )
        return {}
