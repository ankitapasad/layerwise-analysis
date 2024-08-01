import os
import json
import argparse
import scipy
import numpy as np
from tqdm import tqdm
from pathlib import Path

def average_pooling(vecs, n):
    ret = np.cumsum(vecs, axis=0)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

parser = argparse.ArgumentParser()  # o
parser.add_argument('rep_dir', help="path to $save_dir_pth/$model_name/buckeye_$dataset_split_sample1/contextualized/frame_level/")
parser.add_argument('data_sample', help="path to data_samples/buckeye/segmentation/buckeye_$dataset_split.tsv")
parser.add_argument('layer', type=int, help="which layer to use")
parser.add_argument('prominence', type=float, help="hyper-paremter for the peak detection algorithm")
parser.add_argument('dist', choices=['eucli', 'cosine'], default='eucli', help="distance metric")
parser.add_argument('window_size', type=int, help="window size for average pooling")
parser.add_argument('save_dir', help="directory for the segmentation results")
args = parser.parse_args()

# The number of frames in each sentence in args.data_sample
length_file = os.path.join(args.rep_dir, 'n_frames.txt')

stride_sec = 20 / 1000

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


npy_files = os.listdir(args.rep_dir)
for npy_file in npy_files:
    vecs = np.load(os.path.join(args.rep_dir, f'layer_{args.layer}.npy'))
    assert vecs.shape[0] == sum(sentence_frames)
    
vecs_transformed = scipy.stats.zscore(vecs, axis=0)

os.makedirs(args.save_dir, exist_ok=True)

for i in tqdm(range(len(sentence_ids))):
    sentence_id = sentence_ids[i]
    vecs_selected = vecs_transformed[sum(sentence_frames[:i]):sum(sentence_frames[:i+1])]

    if args.dist == "eucli":
        vecs_diff = np.linalg.norm(vecs_selected[1:] - vecs_selected[:-1], axis=1)
    elif args.dist == "cosine":
        vecs_diff = 1-np.sum(vecs_selected[1:]*vecs_selected[:-1], axis=1)/np.linalg.norm(vecs_selected[1:], axis=1)/np.linalg.norm(vecs_selected[:-1], axis=1)
    #     vecs_diff = 2*vecs_diff[1:-1] - vecs_diff[:-2] - vecs_diff[2:]

    vecs_smoothened = average_pooling(vecs_diff, args.window_size)
    peaks, peak_dict = scipy.signal.find_peaks(vecs_smoothened, prominence=args.prominence)
    peak_prominences = peak_dict['prominences']

    if len(peaks) > 0:
        prominences, peaks = zip(*sorted(zip(peak_prominences, peaks), reverse=True))
        prominences, peaks = np.array(prominences), np.array(peaks)
        peaks = peaks + args.window_size//2
    else:
        peaks = []

    with open(f'{args.save_dir}/{sentence_id}.txt', 'w') as f:
        peaks = np.sort(peaks) * 2 + 1 # the frame rate of speech models is 20 ms
        peaks = [0] + peaks.tolist() + [(vecs_selected.shape[0]-1)*2+1]
        for peak_id in range(len((peaks[:-1]))):                     
            f.write(f'{peaks[peak_id]} {peaks[peak_id+1]} xxx\n')
        sentence_id = sentence_ids[i]
        f.write(f'{peaks[peak_id]} {peaks[peak_id+1]} xxx\n')

