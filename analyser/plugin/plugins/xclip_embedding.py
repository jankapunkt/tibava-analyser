import gzip
import os
import html
import ftfy
import regex as re
import numpy as np

import scipy
from scipy import spatial
from analyser.plugins.analyser import AnalyserPlugin, AnalyserPluginManager
from analyser.data import (
    VideoData,
    StringData,
    ScalarData,
    AnnotationData,
    ImageEmbedding,
    TextEmbedding,
    ImageEmbeddings,
    TextEmbeddings,
    generate_id,
)
from analyser.utils import InferenceServer, Backend, Device, VideoDecoder, VideoBatcher
from analyser.utils.imageops import image_resize, image_crop, image_pad
from functools import lru_cache
from typing import Union, List

from sklearn.preprocessing import normalize
import imageio


default_config = {
    "data_dir": "/data/",
    "host": "localhost",
    "port": 6379,
    "video_model_name": "video_xclip",
    "text_model_name": "text_xclip",
    "sim_model_name": "sim_xclip",
    "model_device": "cpu",
    "video_model_file": "/models/xclip/xclip_16_8_video.onnx",
    "text_model_file": "/models/xclip/xclip_16_8_text.onnx",
    "sim_model_file": "/models/xclip/xclip_16_8_sim.onnx",
    "label_file": "/models/xclip/xclip_16_8_sim.onnx",
    "bpe_file": "/models/xclip/bpe_simple_vocab_16e6.txt.gz",
}


@lru_cache()
def bytes_to_unicode():
    """
    Returns list of utf-8 byte and a corresponding list of unicode strings.
    The reversible bpe codes work on unicode strings.
    This means you need a large # of unicode characters in your vocab if you want to avoid UNKs.
    When you're at something like a 10B token dataset you end up needing around 5K for decent coverage.
    This is a signficant percentage of your normal, say, 32K bpe vocab.
    To avoid that, we want lookup tables between utf-8 bytes and unicode strings.
    And avoids mapping to whitespace/control characters the bpe code barfs on.
    """
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))


def get_pairs(word):
    """Return set of symbol pairs in a word.
    Word is represented as tuple of symbols (symbols being variable-length strings).
    """
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs


def basic_clean(text):
    text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    return text.strip()


def whitespace_clean(text):
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text


class SimpleTokenizer(object):
    def __init__(
        self,
        bpe_path: str = "/models/xclip/bpe_simple_vocab_16e6.txt.gz",
        context_length: int = 77,
        truncate: bool = False,
    ):
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        merges = gzip.open(bpe_path).read().decode("utf-8").split("\n")
        merges = merges[1 : 49152 - 256 - 2 + 1]
        merges = [tuple(merge.split()) for merge in merges]
        vocab = list(bytes_to_unicode().values())
        vocab = vocab + [v + "</w>" for v in vocab]
        for merge in merges:
            vocab.append("".join(merge))
        vocab.extend(["<|startoftext|>", "<|endoftext|>"])
        self.encoder = dict(zip(vocab, range(len(vocab))))
        self.decoder = {v: k for k, v in self.encoder.items()}
        self.bpe_ranks = dict(zip(merges, range(len(merges))))
        self.cache = {"<|startoftext|>": "<|startoftext|>", "<|endoftext|>": "<|endoftext|>"}
        self.pat = re.compile(
            r"""<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+""",
            re.IGNORECASE,
        )
        self.context_length = context_length
        self.truncate = truncate

    def bpe(self, token):
        if token in self.cache:
            return self.cache[token]
        word = tuple(token[:-1]) + (token[-1] + "</w>",)
        pairs = get_pairs(word)

        if not pairs:
            return token + "</w>"

        while True:
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float("inf")))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except:
                    new_word.extend(word[i:])
                    break

                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)
        word = " ".join(word)
        self.cache[token] = word
        return word

    def encode(self, text):
        bpe_tokens = []
        text = whitespace_clean(basic_clean(text)).lower()
        for token in re.findall(self.pat, text):
            token = "".join(self.byte_encoder[b] for b in token.encode("utf-8"))
            bpe_tokens.extend(self.encoder[bpe_token] for bpe_token in self.bpe(token).split(" "))
        return bpe_tokens

    def decode(self, tokens):
        text = "".join([self.decoder[token] for token in tokens])
        text = bytearray([self.byte_decoder[c] for c in text]).decode("utf-8", errors="replace").replace("</w>", " ")
        return text

    def tokenize(self, texts: Union[str, List[str]]):
        if isinstance(texts, str):
            texts = [texts]

        sot_token = self.encoder["<|startoftext|>"]
        eot_token = self.encoder["<|endoftext|>"]
        all_tokens = [[sot_token] + self.encode(text) + [eot_token] for text in texts]

        result = np.zeros((len(all_tokens), self.context_length), dtype=int)

        for i, tokens in enumerate(all_tokens):
            if len(tokens) > self.context_length:
                if self.truncate:
                    tokens = tokens[: self.context_length]
                    tokens[-1] = eot_token
                else:
                    raise RuntimeError(f"Input {texts[i]} is too long for context length {self.context_length}")
            result[i, : len(tokens)] = np.array(tokens, dtype=int)

        return result


img_embd_parameters = {"fps": 2, "crop_size": [224, 224], "batch_size": 8}

img_embd_requires = {
    "video": VideoData,
}

img_embd_provides = {
    "image_features": ImageEmbeddings,
    "video_features": ImageEmbeddings,
}


@AnalyserPluginManager.export("x_clip_video_embedding")
class XClipVideoEmbedding(
    AnalyserPlugin,
    config=default_config,
    parameters=img_embd_parameters,
    version="0.1",
    requires=img_embd_requires,
    provides=img_embd_provides,
):
    def __init__(self, config=None):
        super().__init__(config)
        self.host = self.config["host"]
        self.port = self.config["port"]
        self.model_name = self.config["video_model_file"]
        self.model_device = self.config["model_device"]
        self.model_file = self.config["video_model_file"]

        self.server = InferenceServer(
            model_file=self.model_file,
            model_name=self.model_name,
            host=self.host,
            port=self.port,
            backend=Backend.ONNX,
            device=self.model_device,
            inputs=["image"],
            outputs=["video_features", "image_features"],
        )

    def preprocess(self, img, resize_size, crop_size):
        converted = image_resize(image_pad(img), size=crop_size)
        converted = (converted - np.asarray([123.675, 116.28, 103.53])) / np.asarray([58.395, 57.12, 57.375])
        converted = converted.astype(np.float32)
        converted = np.transpose(converted, [2, 0, 1])
        return converted

    # img_norm_cfg = dict(
    #     mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False)
    #         return converted

    def call(self, inputs, parameters, callbacks=None):
        print("1", flush=True)
        video_features = []
        image_features = []
        video_decoder = VideoBatcher(
            VideoDecoder(path=inputs["video"].path, fps=parameters.get("fps")), batch_size=parameters.get("batch_size")
        )
        print("2", flush=True)
        num_frames = (video_decoder.duration() * video_decoder.fps()) // parameters.get("batch_size")
        print(f"3 {num_frames}", flush=True)
        for i, frame in enumerate(video_decoder):
            self.update_callbacks(callbacks, progress=i / num_frames)
            img_id = generate_id()
            imgs = frame.get("frame")
            print(f"3 {imgs.shape}", flush=True)
            preprocessed_imgs = []
            for img in imgs:
                img = self.preprocess(img, parameters.get("resize_size"), parameters.get("crop_size"))
                preprocessed_imgs.append(img)
            preprocessed_imgs = np.expand_dims(np.stack(preprocessed_imgs), 0)
            print(f"3 {preprocessed_imgs.shape}", flush=True)
            # imageio.imwrite(os.path.join(self.config.get("data_dir"), f"test_{i}.jpg"), img)
            result = self.server({"image": preprocessed_imgs}, ["video_features", "image_features"])
            print(result, flush=True)
            video_features.append(
                ImageEmbedding(
                    embedding=result["video_features"],
                    image_id=img_id,
                    time=frame.get("time")[0],
                    delta_time=1 / parameters.get("fps"),
                )
            )

            image_features.append(
                ImageEmbedding(
                    embedding=result["image_features"],
                    image_id=img_id,
                    time=frame.get("time")[0],
                    delta_time=1 / parameters.get("fps"),
                )
            )

        self.update_callbacks(callbacks, progress=1.0)
        return {
            "image_features": ImageEmbeddings(embeddings=image_features),
            "video_features": ImageEmbeddings(embeddings=video_features),
        }


text_embd_parameters = {
    "search_term": "",
}
text_embd_requires = {}

text_embd_provides = {
    "embeddings": TextEmbeddings,
}


@AnalyserPluginManager.export("x_clip_text_embedding")
class XClipTextEmbedding(
    AnalyserPlugin,
    config=default_config,
    parameters=text_embd_parameters,
    version="0.1",
    requires=text_embd_requires,
    provides=text_embd_provides,
):
    def __init__(self, config=None):
        super().__init__(config)
        self.host = self.config["host"]
        self.port = self.config["port"]
        self.model_name = self.config["text_model_name"]
        self.model_device = self.config["model_device"]
        self.model_file = self.config["text_model_file"]
        self.bpe_path = self.config["bpe_file"]
        self.tokenizer = SimpleTokenizer(self.bpe_path)

        self.server = InferenceServer(
            model_file=self.model_file,
            model_name=self.model_name,
            host=self.host,
            port=self.port,
            backend=Backend.ONNX,
            device=self.model_device,
            inputs=["text"],
            outputs=["text_features"],
        )

    def preprocess(self, text):
        # tokenize text

        tokenized = self.tokenizer.tokenize(text)
        return tokenized

    def call(self, inputs, parameters, callbacks=None):
        text_id = generate_id()
        text = self.preprocess(parameters["search_term"])
        # print(text.shape)

        result = self.server({"text": text}, ["text_features"])

        self.update_callbacks(callbacks, progress=1.0)
        return {
            "embeddings": TextEmbeddings(
                embeddings=[
                    TextEmbedding(text_id=text_id, text=parameters["search_term"], embedding=result["text_features"][0])
                ]
            )
        }


prob_parameters = {
    "search_term": "",
}

prob_requires = {
    "image_features": ImageEmbeddings,
    "video_features": ImageEmbeddings,
}

prob_provides = {
    "probs": ScalarData,
}


@AnalyserPluginManager.export("x_clip_probs")
class XClipProbs(
    AnalyserPlugin,
    config=default_config,
    parameters=prob_parameters,
    version="0.1",
    requires=prob_requires,
    provides=prob_provides,
):
    def __init__(self, config=None):
        super().__init__(config)
        self.host = self.config["host"]
        self.port = self.config["port"]
        self.model_device = self.config["model_device"]
        self.text_model_name = self.config["text_model_name"]
        self.text_model_file = self.config["text_model_file"]
        self.sim_model_name = self.config["sim_model_name"]
        self.sim_model_file = self.config["sim_model_file"]
        self.bpe_path = self.config["bpe_file"]
        self.tokenizer = SimpleTokenizer(self.bpe_path)

        self.text_server = InferenceServer(
            model_file=self.text_model_file,
            model_name=self.text_model_name,
            host=self.host,
            port=self.port,
            backend=Backend.ONNX,
            device=self.model_device,
            inputs=["text"],
            outputs=["text_features"],
        )

        self.sim_server = InferenceServer(
            model_file=self.sim_model_file,
            model_name=self.sim_model_name,
            host=self.host,
            port=self.port,
            backend=Backend.ONNX,
            device=self.model_device,
            inputs=["text_features", "video_features", "image_features"],
            outputs=["probs", "scale"],
        )

    def preprocess(self, text):
        # tokenize text
        tokenized = self.tokenizer.tokenize(text)
        return tokenized

    def call(self, inputs, parameters, callbacks=None):
        probs = []
        time = []
        delta_time = None
        image_features = inputs["image_features"]
        video_features = inputs["video_features"]

        text = self.preprocess(parameters["search_term"])
        result = self.text_server({"text": text}, ["text_features"])

        text_embedding = normalize(result["text_features"])

        neg_text = self.preprocess("Not " + parameters["search_term"])
        neg_result = self.text_server({"text": neg_text}, ["text_features"])

        neg_text_embedding = normalize(neg_result["text_features"])

        text_embedding = np.concatenate([text_embedding, neg_text_embedding], axis=0)
        for image_feature, video_feature in zip(image_features.embeddings, video_features.embeddings):
            print(f"#### {text_embedding.shape}", flush=True)
            print(f"#### {video_feature.embedding.shape}", flush=True)
            print(f"#### {image_feature.embedding.shape}", flush=True)
            result = self.sim_server(
                {
                    "text_features": text_embedding,
                    "video_features": video_feature.embedding,
                    "image_features": image_feature.embedding,
                },
                ["probs", "scale"],
            )
            # result = 100 * text_embedding @ embedding.embedding.T

            prob = scipy.special.softmax(result["probs"], axis=0)

            # sim = 1 - spatial.distance.cosine(embedding.embedding, text_embedding)
            probs.append(prob[0, 0])
            time.append(image_feature.time)
            delta_time = image_feature.delta_time

        self.update_callbacks(callbacks, progress=1.0)
        return {
            "probs": ScalarData(y=np.array(probs), time=time, delta_time=delta_time, name="image_text_similarities")
        }


anno_parameters = {
    "threshold": 0.5,
}


anno_requires = {
    "embeddings": ImageEmbeddings,
    "search_term": StringData,
}


# anno_provides = {
#     "annotation": AnnotationData,
# }

# @AnalyserPluginManager.export("xclip_annotation")
# class ImageTextAnnotation(
#     Plugin,
#     config=default_config,
#     parameters=anno_parameters,
#     version="0.1",
#     requires=anno_requires,
#     provides=anno_provides,
# ):
#     def __init__(self, config=None):
#         super().__init__(config)

#     def call(self, inputs, parameters, callbacks=None):
#         # TODO
#         return {}
