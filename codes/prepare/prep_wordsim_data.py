"""
Prepare data for feature extraction for the word similarity tasks
"""
import fire
import numpy as np
import os

curr_dir = os.path.dirname(os.path.realpath(__file__))
import sys

sys.path.insert(0, os.path.join(curr_dir, ".."))
from utils import read_lst, save_dct, write_to_file, load_dct


def filter_data(
    words_of_interest, wrd_lst, wrd_cnt_dct, task_dct, fname, wrd_pairs, thresh
):
    """
    Output:
        Task rw: 1111 0.55 pairs retained
        Task semeval17: 269 0.54 pairs retained
        Task wordsim353-sim: 184 0.9 pairs retained
        Task mturk-771: 734 0.95 pairs retained
        Task mturk-287: 263 0.92 pairs retained
        Task mc-30: 30 1.0 pairs retained
        Task wordsim353-rel: 225 0.89 pairs retained
        Task rg-65: 65 1.0 pairs retained
        Task yp-130: 122 0.94 pairs retained
        Task men: 2824 0.94 pairs retained
        Task simlex999: 992 0.99 pairs retained
        Task verb-143: 118 0.91 pairs retained
        Task simverb-3500: 3437 0.98 pairs retained
    """
    tot_num_pairs = len(wrd_pairs)
    task_name = fname.split(".")[0]
    for item in wrd_pairs:
        if "simverb-3500" not in fname:
            _, w1, w2, score = item.split(",")
        else:
            _, score, w1, w2, _ = item.split(",")
        if w1 != "":
            if "men" in fname:
                w1 = w1.split("-")[0]
                w2 = w2.split("-")[0]
            score = float(score)
            if w1 in wrd_lst and w2 in wrd_lst:
                if wrd_cnt_dct[w1] > thresh and wrd_cnt_dct[w2] > thresh:
                    _ = task_dct.setdefault(task_name, [])
                    task_dct[task_name].append((w1, w2, score))
                    words_of_interest.extend([w1, w2])
    print(
        f"Task {task_name}: {len(task_dct[task_name])} {(np.round(len(task_dct[task_name])/tot_num_pairs, 2))} pairs retained"
    )


def prepare_task_data(wordsim_data_dir, alignment_data_dir, save_dir, thresh=0):
    """
    Filter and reformat the wordsim data to include words that are available in the training data

    wordsim_data_dir: Path to the cloned git repo for word similarity tasks
    alignment_data_dir: Path to where the librispeech alignments are stored
    thresh: a threshold on the word occurence count in the librispeech train data
    """
    wordsim_data_subdir = os.path.join(
        wordsim_data_dir, "word-similarity/monolingual/en"
    )
    csv_fnames = os.listdir(wordsim_data_subdir)
    wrd_lst = read_lst(os.path.join(alignment_data_dir, "word.lst"))
    wrd_cnt_dct = load_dct(os.path.join(alignment_data_dir, "word_count.json"))
    words_of_interest = []
    task_dct = {}
    for fname in csv_fnames:
        all_pairs = read_lst(os.path.join(wordsim_data_subdir, fname))[1:]
        filter_data(
            words_of_interest, wrd_lst, wrd_cnt_dct, task_dct, fname, all_pairs, thresh
        )
        words_of_interest = list(set(words_of_interest))
    cnt_lst = [wrd_cnt_dct[wrd] for wrd in words_of_interest]
    save_dct(os.path.join(save_dir, f"wordsim_tasks_thresh{thresh}.json"), task_dct)
    write_to_file(
        "\n".join(words_of_interest),
        os.path.join(save_dir, f"words_of_interest_thresh{thresh}.lst"),
    )


if __name__ == "__main__":
    fire.Fire(prepare_task_data)
