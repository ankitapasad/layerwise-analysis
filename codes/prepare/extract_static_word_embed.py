import fire
from glob import glob
import numpy as np
import os
from pathlib import Path
import time
from tqdm import tqdm

curr_dir = os.path.dirname(os.path.realpath(__file__))
import sys

sys.path.insert(0, os.path.join(curr_dir, ".."))
from model_utils import ModelLoader, FeatExtractor
from utils import read_lst, save_dct, load_dct


def save_rep(
    model_name,
    ckpt_pth,
    save_dir,
    utt_id_fn,
    model_type="pretrained",
    dict_fn=None,
    pckg_dir=None,
):
    """
    Extract layer-wise representations from the model

    ckpt_pth: path to the model checkpoint
    save_dir: directory where the representations are saved
    utt_id_fn: identifier for utterances
    model_type: pretrained or finetuned
    dict_fn: path to dictionary file in case of finetuned models
    """
    rep_type = "contextualized"
    model_obj = ModelLoader(ckpt_pth, model_type, pckg_dir, dict_fn)
    encoder, task_cfg = getattr(model_obj, model_name.split("_")[0])()

    Path(save_dir).mkdir(exist_ok=True, parents=True)
    utt_id_lst = read_lst(utt_id_fn)
    label_lst = read_lst(utt_id_fn.replace("word_segments", "labels"))
    rep_dct = {}  # word to list of rep mapping
    idx = 0
    start_time = time.time()
    for item in tqdm(utt_id_lst):
        utt_id, wav_fn, start, end = item.split(",")
        time_stamp_lst = [(start, end, label_lst[idx])]
        extract_obj = FeatExtractor(
            encoder,
            utt_id,
            wav_fn,
            "contextualized",
            model_name,
            task_cfg=task_cfg,
            offset=False,
            mean_pooling=True,
        )
        getattr(extract_obj, model_name.split("_")[0])()
        extract_obj.extract_contextualized_rep(rep_dct, time_stamp_lst)
        idx += 1
    wrd_to_idx = {}
    word_embedding = {}

    for idx, label in enumerate(label_lst):
        _ = wrd_to_idx.setdefault(label, [])
        wrd_to_idx[label].append(idx)
        if label not in word_embedding:
            word_embedding[label] = {}

    for layer_num in range(len(rep_dct)):
        rep_mat = np.concatenate(rep_dct[layer_num], 0)
        for wrd, idx_lst in wrd_to_idx.items():
            word_embedding[wrd][layer_num] = np.mean(rep_mat[idx_lst], 0)

    subset_id = utt_id_fn.split("_")[-1].split(".")[0]
    save_fn = f"{subset_id}.pkl"
    save_dct(os.path.join(save_dir, save_fn), word_embedding)
    print("%d word embeddings saved to %s" % (len(wrd_to_idx), save_dir))
    print("Time required: %.1f mins" % ((time.time() - start_time) / 60))


def combine_embeddings(embedding_dir):
    """
    Combine all extracted embeddings into a single pkl file
    """
    all_files = glob(os.path.join(embedding_dir, "*.pkl"))
    all_wrd_embed_dct = {}
    for file_idx in tqdm(range(len(all_files))):
        fname = os.path.join(embedding_dir, f"{file_idx}.pkl")
        wrd_embed_dct = load_dct(fname)
        for wrd, layer_dct in wrd_embed_dct.items():
            for layer_num, embed_vec in layer_dct.items():
                _ = all_wrd_embed_dct.setdefault(layer_num, {})
                assert wrd not in all_wrd_embed_dct[layer_num]
                all_wrd_embed_dct[layer_num][wrd] = embed_vec
        os.remove(fname)
    for layer_num, wrd_embedding_map in all_wrd_embed_dct.items():
        save_dct(
            os.path.join(embedding_dir, f"layer{layer_num}.pkl"), wrd_embedding_map
        )


if __name__ == "__main__":
    fire.Fire(
        {
            "extract": save_rep,
            "combine": combine_embeddings,
        }
    )
