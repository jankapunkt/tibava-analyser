import argparse
import imageio.v3 as iio
import logging
import sys

from analyser.client import AnalyserClient
from analyser.data import DataManager, ImageData, ImagesData, generate_id, create_data_path


def parse_args():
    parser = argparse.ArgumentParser(description="")

    parser.add_argument("-v", "--verbose", action="store_true", help="verbose output")
    parser.add_argument("--input_path", default="/media/test.mp4", help="path to input video .mp4")
    parser.add_argument("--query_images", default=["/media/test.jpg"], nargs="+", help="path to query image .jpg")
    parser.add_argument("--output_path", default="/media", help="path to output folder")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    level = logging.ERROR
    if args.verbose:
        level = logging.INFO

    logging.basicConfig(format="%(asctime)s %(levelname)s: %(message)s", datefmt="%d-%m-%Y %H:%M:%S", level=level)
    client = AnalyserClient("localhost", 50051)

    """
    UPLOAD VIDEO
    """
    logging.info(f"Start uploading")
    data_id = client.upload_file(args.input_path)
    logging.info(f"Upload done: {data_id}")

    """
    FACE DETECTION FROM TARGET VIDEO
    """
    job_id = client.run_plugin("insightface_video_detector", [{"id": data_id, "name": "video"}], [])
    logging.info(f"Job insightface_video_detector started: {job_id}")

    result = client.get_plugin_results(job_id=job_id)
    if result is None:
        logging.error("Job is crashing")
        return

    target_kpss_id = None
    for output in result.outputs:
        if output.name == "kpss":
            target_kpss_id = output.id

    target_kpss_data = client.download_data(target_kpss_id, args.output_path)
    logging.info(target_kpss_data)

    """
    FACIAL FEATURE EXTRACTION FROM TARGET VIDEO
    """
    job_id = client.run_plugin(
        "insightface_video_feature_extractor",
        [
            {"id": data_id, "name": "video"},
            {"id": target_kpss_id, "name": "kpss"},
        ],
        [],
    )
    logging.info(f"Job insightface_video_feature_extractor started: {job_id}")

    result = client.get_plugin_results(job_id=job_id)
    target_features_id = None
    for output in result.outputs:
        if output.name == "features":
            target_features_id = output.id
    logging.info(target_features_id)

    """
    UPLOAD QUERY IMAGE(S)
    """
    manager = DataManager()
    client = AnalyserClient("localhost", 50051, manager=manager)
    images = []
    logging.info(f"Read query images")
    for image_path in args.query_images:
        image_id = generate_id()
        output_path = create_data_path(manager.data_dir, image_id, "jpg")
        image = iio.imread(image_path)
        iio.imwrite(output_path, image)
        images.append(ImageData(id=image_id, ext="jpg"))

    data = ImagesData(images=images)
    logging.info("Start uploading query images")
    query_image_ids = client.upload_data(data)
    logging.info(f"Upload done: {query_image_ids}")

    """
    FACE DETECTION FROM QUERY IMAGE(S)
    """
    job_id = client.run_plugin("insightface_image_detector", [{"id": query_image_ids, "name": "images"}], [])
    logging.info(f"Job insightface_image_detector started: {job_id}")

    result = client.get_plugin_results(job_id=job_id)
    if result is None:
        logging.error("Job is crashing")
        return

    query_kpss_id = None
    for output in result.outputs:
        if output.name == "kpss":
            query_kpss_id = output.id

    query_kpss_data = client.download_data(query_kpss_id, args.output_path)
    logging.info(query_kpss_data)

    """
    FACIAL FEATURE EXTRACTION FROM QUERY IMAGE(S)
    """
    job_id = client.run_plugin(
        "insightface_image_feature_extractor",
        [
            {"id": query_image_ids, "name": "images"},
            {"id": query_kpss_id, "name": "kpss"},
        ],
        [],
    )
    logging.info(f"Job insightface_image_feature_extractor started: {job_id}")

    result = client.get_plugin_results(job_id=job_id)
    query_features_id = None
    for output in result.outputs:
        if output.name == "features":
            query_features_id = output.id

    logging.info(query_features_id)

    """
    FACE SIMILARITY
    """
    job_id = client.run_plugin(
        "cosine_similarity",
        [{"id": target_features_id, "name": "target_features"}, {"id": query_features_id, "name": "query_features"}],
        [],
    )
    logging.info(f"Job cosine_similarity started: {job_id}")

    result = client.get_plugin_results(job_id=job_id)
    for output in result.outputs:
        if output.name == "probs":
            similarities_id = output.id

    similarities_data = client.download_data(similarities_id, args.output_path)
    logging.info(similarities_data)

    return 0


if __name__ == "__main__":
    sys.exit(main())
