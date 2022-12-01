import fire
import numpy as np
import os
from tqdm import tqdm

curr_dir = os.path.dirname(os.path.realpath(__file__))
import sys

sys.path.insert(0, os.path.join(curr_dir, ".."))
from utils import save_dct, read_lst


def save_as_dct(embed_fn, out_fn):
    embed_txt = read_lst(embed_fn)
    embed_dct = {}
    for line in tqdm(embed_txt):
        word = line.split(" ")[0]
        embed_array = np.array(list(map(float, line.split(" ")[1:])))
        embed_dct[word] = embed_array
    save_dct(out_fn, embed_dct)


if __name__ == "__main__":
    fire.Fire()
