import argparse
import logging
import os
import sys
import warnings

import gensim
import numpy as np
import pandas as pd
from gensim import models
from gensim.test.utils import datapath

from alignment import smart_procrustes_re_align_gensim

verbose_format = "%(asctime)s %(name)s %(levelname)s %(message)s"
warnings.simplefilter("ignore")
logging.basicConfig(format=verbose_format, level=logging.WARNING, stream=sys.stdout)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def cos_sim(v1: np.ndarray, v2: np.ndarray) -> float:
    """calculate cosine similarity

    Args:
        v1: vector 1
        v2: vector 2

    Returns:
        float: cosine similarity
    """
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


def load_keyvector(file_path: str) -> models.keyedvectors:
    """load KeyedVectors

    Args:
        file_path (str): file path of wv

    Returns:
        KeyedVectors: wv
    """
    _, ext = os.path.splitext(file_path)
    if ext == ".model":
        model = gensim.models.Word2Vec.load(file_path)
        wv = model.wv
        del model
        return wv
    elif ext == ".bin":
        wv = gensim.models.KeyedVectors.load_word2vec_format(
            datapath(file_path), binary=True
        )
        return wv
    elif ext == ".txt" or ext == ".vec":
        wv = gensim.models.KeyedVectors.load_word2vec_format(
            datapath(file_path), binary=False
        )
        return wv
    elif ext == ".kv":
        wv = gensim.models.KeyedVectors.load(file_path, mmap="r")
        return wv
    else:
        raise ValueError(f"Cant load extension {ext} data")


def calculate_semantic_shift_stability(
    wv1_path: str, wv2_path: str, save_words: bool = False
) -> float:
    logger.info(f"loading {wv1_path} ...")
    wv1 = load_keyvector(wv1_path)
    logger.info(f"loading {wv2_path} ...")
    wv2 = load_keyvector(wv2_path)

    re_aligned_wv1 = smart_procrustes_re_align_gensim(wv2, wv1)
    re_aligned_wv2 = smart_procrustes_re_align_gensim(wv1, wv2)

    stab = {}
    for vocab in re_aligned_wv1.vocab.keys():
        stab[vocab] = (
            cos_sim(re_aligned_wv1.wv[vocab], wv1.wv[vocab])
            + cos_sim(re_aligned_wv2.wv[vocab], wv2.wv[vocab])
        ) / 2
    stab = sorted(stab.items(), key=lambda x: x[1])

    df = pd.DataFrame(stab)
    df.columns = ["word", "stability"]

    if save_words:
        df.to_csv("semantic_shift_stability_per_word.csv", index=False)

    return df["stability"].mean()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--wv1_path", type=str, required=True, help="w2v file path 1")
    parser.add_argument("--wv2_path", type=str, required=True, help="w2v file path 2")
    args = parser.parse_args()

    semantic_shift_stability = calculate_semantic_shift_stability(
        args.wv1_path, args.wv2_path
    )
    logger.info(f"semantic shift stability: {semantic_shift_stability}")
