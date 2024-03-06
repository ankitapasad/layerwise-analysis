import os
import json
import argparse
import scipy
import numpy as np
import textgrids
from pathlib import Path

def get_word_alignment(chosen_fnames, chosen_sent_ids, data_dir, alignment_dir, sent_frames):
    """
    Get the word-level alignments for the chosen sentences
    """
    alignments = []
    def get_segment_idx(start_time, end_time, len_utt):
        start_id = int(np.ceil(float(start_time) / stride_sec))
        end_id = int(np.ceil(float(end_time) / stride_sec))
        if end_id <= start_id:
            end_id = start_id + 1
        if end_id > len_utt:
            end_id = len_utt
        assert end_id > start_id

        return [start_id, end_id]
    
    def read_grid(grid, max_frame):
        data_lst = []
        for item in grid:
            label = item.text
            if label:  # check that it is non-empty
                start = str(item.xmin)
                end = str(item.xmax)
                if label == "spn" or "<" in label: # special cases not handled
                    return None
                elif label == "sp":
                    label = "sil"
                    
                out = get_segment_idx(start, end, max_frame) + [item.xmin, item.xmax, label]
                data_lst.append(out)
        return data_lst

    for sent_id, fname, n_frames in zip(chosen_sent_ids, chosen_fnames, sent_frames):
        relpath = os.path.relpath(fname, data_dir)
        gridpath = (Path(alignment_dir)/relpath).with_suffix(".TextGrid")
        grid = textgrids.TextGrid(gridpath)
        
        phones = read_grid(grid["phones"], n_frames)
        words = read_grid(grid["words"], n_frames)
        if phones is None or words is None:
            continue

        alignments.append((words, phones))
    return alignments

def average_pooling(vecs, n):
    ret = np.cumsum(vecs, axis=0)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def f1_score(seqs_gt, seqs_pred, stride_sec, tolerance):
    total_tp = 0
    total_fp = 0
    total_fn = 0
    for seq_gt, seq_pred in zip(seqs_gt, seqs_pred):
        seq_pred = seq_pred * stride_sec
        tp = 0
        fp = 0
        fn = 0
        for s_p in seq_pred:
            add_tp = False
            for s_g in seq_gt:
                if s_g - tolerance <= s_p and s_p <= s_g + tolerance:
                    add_tp = True
                    break
            if add_tp:
                tp += 1
            else:
                fp += 1
        for s_g in seq_gt:
            add_fn = True
            for s_p in seq_pred:
                if s_g - tolerance <= s_p and s_p <= s_g + tolerance:
                    add_fn = False
                    break
            if add_fn:
                fn += 1
        total_tp += tp
        total_fp += fp
        total_fn += fn
    if total_tp == 0:
        return 0, 0, 0
    precision = total_tp/(total_tp+total_fp)
    recall = total_tp/(total_tp+total_fn)
    f1 = 2*precision*recall/(precision+recall)
    return precision, recall, f1

parser = argparse.ArgumentParser()  # o
parser.add_argument('rep_dir', help="path to $save_dir_pth/$model_name/librispeech_$dataset_split_sample1/contextualized/frame_level/")
parser.add_argument('data_sample', help="path to data_samples/librispeech/frame_level/500_ids_sample1_dev-clean.tsv")
parser.add_argument('data_dir', help="path to LibriSpeech Data")
parser.add_argument('alignment_dir', help="path to alignment data")
parser.add_argument('--save_path', default="", required=False) 
args = parser.parse_args()

# The number of frames in each sentence in args.data_sample
length_file = os.path.join(args.rep_dir, 'n_frames.txt')

stride_sec = 20 / 1000
tolerance = 0.02
layers = list(range(1, 25))

# Grid search over these hyper-parameters
prominence_values = [0, 0.00001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 2, 4, 6, 8, 10]
dist_metrics = ["eucli", "cosine"]
window_sizes = [1, 2, 4, 6, 8, 16]

sentence_frames = []
with open(length_file, 'r') as f:
    for line in f:
        sentence_frames.append(int(line.strip('\n')))
        
sentence_ids = []
fnames = []
with open(args.data_sample, 'r') as f:
    for line in f:
        sentence_id, fname = line.strip('\n').split('\t')
        sentence_ids.append(sentence_id)
        fnames.append(fname)

alignments = get_word_alignment(fnames, sentence_ids, args.data_dir, args.alignment_dir, sentence_frames)

best_results = {}
for layer in layers:
    npy_files = os.listdir(args.rep_dir)
    for npy_file in npy_files:
        if "npy" not in npy_file:
            continue
        if layer != int(npy_file.split('.')[0].split('_')[1]):
            continue
        vecs = np.load(os.path.join(args.rep_dir, npy_file))
        assert vecs.shape[0] == sum(sentence_frames)
    
    layer_best_f1 = 0
    for prominence in prominence_values:
        for dist in dist_metrics:
            for window_size in window_sizes:
                vecs_transformed = scipy.stats.zscore(vecs, axis=0)

                all_boundaries = []
                all_peaks = []
                for i in range(len(sentence_ids)):
                    word_alignment, phone_alignment = alignments[i]
                    alignment = word_alignment
                    boundaries = sorted(list(set([time for unit in alignment for time in unit[2:4]])))
                    all_boundaries.append(boundaries)

                    sentence_id = sentence_ids[i]
                    vecs_selected = vecs_transformed[sum(sentence_frames[:i]):sum(sentence_frames[:i+1])]
                        
                    if dist == "eucli":
                        vecs_diff = np.linalg.norm(vecs_selected[1:] - vecs_selected[:-1], axis=1)
                    elif dist == "cosine":
                        vecs_diff = 1-np.sum(vecs_selected[1:]*vecs_selected[:-1], axis=1)/np.linalg.norm(vecs_selected[1:], axis=1)/np.linalg.norm(vecs_selected[:-1], axis=1)
                    #     vecs_diff = 2*vecs_diff[1:-1] - vecs_diff[:-2] - vecs_diff[2:]
                        
                    vecs_smoothened = average_pooling(vecs_diff, window_size)
                    peaks, peak_dict = scipy.signal.find_peaks(vecs_smoothened, prominence=prominence)
                    peak_prominences = peak_dict['prominences']
                    if peak_prominences.size == 0:
                        print(f"no peaks detected for sentence {i}")
                        continue

                    prominences, peaks = zip(*sorted(zip(peak_prominences, peaks), reverse=True))
                    prominences, peaks = np.array(prominences), np.array(peaks)
                    peaks = peaks + window_size//2
                    all_peaks.append(np.sort(peaks))
                    
                precision, recall, f1 = f1_score(all_boundaries, all_peaks, stride_sec, tolerance)
                if f1 > layer_best_f1:
                    best_dist = dist
                    best_window_size = window_size
                    best_prominence = prominence
                    layer_best_f1 = f1
                    layer_best_precision = precision
                    layer_best_recall = recall
    # print(best_dist, best_window_size, best_prominence)
    # print(layer_best_f1, layer_best_precision, layer_best_recall)
    best_results[layer] = {
        "distance metrics": best_dist, 
        "avg pooling window size": best_window_size,
        "prominence": best_prominence,
        "f1 score": layer_best_f1
    }

if args.save_path:
    with open(args.save_path, 'w') as f:
        f.write(json.dumps(best_results, indent=4))

print(best_results)
