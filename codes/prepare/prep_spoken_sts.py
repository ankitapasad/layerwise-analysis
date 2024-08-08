"""
Download spoken STS from HuggingFace and pre-process the files
"""

from datasets import load_dataset
from fire import Fire
import numpy as np
import os
import soundfile as sf
from tqdm import tqdm

curr_dir = os.path.dirname(os.path.realpath(__file__))
import sys

sys.path.insert(0, os.path.join(curr_dir, ".."))
from utils import read_lst, load_dct, write_to_file, save_dct

def save_audio(sample, audio_dir, tsv_lst, num_secs, idx, pair_id_to_idx):
    wav_dir = os.path.join(audio_dir, sample['task'], sample['subtask'])
    os.makedirs(wav_dir, exist_ok=True)
    for sfx in ['a', 'b']:
        audio = sample[f'audio_{sfx}']['array']
        fs = sample[f'audio_{sfx}']['sampling_rate']
        pair_id = sample['pair_id']
        spk_id = sample['speaker_id']
        utt_id = f'{pair_id}_{spk_id}_{sfx}'
        task_name = sample['task']
        subtask_name = sample['subtask']
        pair_idx = sample['pair_id']
        key = f'{task_name}_{subtask_name}_{pair_idx}'
        _ = pair_id_to_idx.setdefault(key, {})
        _ = pair_id_to_idx[key].setdefault(sfx, [])
        pair_id_to_idx[key][sfx].append(idx)
        idx += 1 
        wav_fn = os.path.join(wav_dir, f'{utt_id}.wav')
        if not os.path.exists(wav_fn):
            sf.write(wav_fn, audio, fs)
        tsv_lst.append("\t".join([utt_id, wav_fn]))
        num_secs[0] += len(audio)/fs
    return idx

def save_gt(sample, gt_dct):
    task_name = sample['task']
    subtask_name = sample['subtask']
    pair_idx = sample['pair_id']
    gt_score = sample['similarity']
    pair_id = f'{task_name}_{subtask_name}_{pair_idx}'
    if pair_id in gt_dct:
        assert gt_dct[pair_id] == gt_score
    else:
        gt_dct[pair_id] = gt_score

def save_all_pairs(data_dir, pair_id_to_idx):
    all_pair_idx_dct = {}
    for pair_id in pair_id_to_idx:
        lst_a = pair_id_to_idx[pair_id]['a']
        lst_b = pair_id_to_idx[pair_id]['b']
        all_pair_idx_dct[pair_id] = []
        for spk_a_idx in lst_a:
            for spk_b_idx in lst_b:
                all_pair_idx_dct[pair_id].append([spk_a_idx, spk_b_idx])
    save_dct(os.path.join(data_dir, "all_pairs_idx.json"), all_pair_idx_dct)

def main(data_dir, dur_thresh=10000):
    sts_obj = load_dataset("juice500/spoken_sts", cache_dir=data_dir)
    data_dir = os.path.join(data_dir, "spoken_sts")
    audio_dir = os.path.join(data_dir, "audio")
    sample_data_dir = os.path.join("data_samples", "spoken_sts", "utt_level")
    tsv_lst = []
    gt_dct = {}
    file_id = 0
    idx = 0
    pair_id_to_idx = {}
    num_secs = [0]
    os.makedirs(sample_data_dir, exist_ok=True)
    for sample in tqdm(sts_obj['test']):
        idx = save_audio(sample, audio_dir, tsv_lst, num_secs, idx, pair_id_to_idx)
        save_gt(sample, gt_dct)
        if num_secs[0] > dur_thresh:
            write_to_file(
                "\n".join(tsv_lst), os.path.join(sample_data_dir, f"split{file_id}.tsv")
            )
            print(f"split {file_id+1} saved at {sample_data_dir}: {np.round(num_secs[0]/3600, 2)} hours")
            file_id += 1
            tsv_lst = []
            num_secs = [0]
    if len(tsv_lst) > 0:
        write_to_file(
            "\n".join(tsv_lst), os.path.join(sample_data_dir, f"split{file_id}.tsv")
        )
        print(f"split {file_id+1} saved at {sample_data_dir}: {np.round(num_secs[0]/3600, 2)} hours")
    save_dct(os.path.join(data_dir, 'all_gt.json'), gt_dct)
    save_all_pairs(data_dir, pair_id_to_idx)

if __name__ == "__main__":
    Fire(main)