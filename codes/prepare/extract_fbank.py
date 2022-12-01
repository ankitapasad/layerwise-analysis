import fire
import librosa
import numpy as np
import os
import scipy
import time
from tqdm import tqdm

from multiprocessing import cpu_count
from concurrent.futures import ProcessPoolExecutor
from functools import partial

curr_dir = os.path.dirname(os.path.realpath(__file__))
import sys

sys.path.insert(0, os.path.join(curr_dir, ".."))
from utils import read_lst, format_time


def preemphasis(x, preemph):
    return scipy.signal.lfilter([1, -preemph], [1], x)


def process_wav(
    wav_path,
    out_path,
    sr=16000,
    preemph=0.97,
    n_fft=2048,
    n_mels=80,
    hop_length=160,
    win_length=400,
    fmin=50,
    top_db=80,
    offset=0.0,
    duration=None,
):
    wav, _ = librosa.load(wav_path, sr=sr, offset=offset, duration=duration)
    wav = wav / np.abs(wav).max() * 0.999
    mel = librosa.feature.melspectrogram(
        y=preemphasis(wav, preemph),
        sr=sr,
        n_fft=n_fft,
        n_mels=n_mels,
        hop_length=hop_length,
        win_length=win_length,
        fmin=fmin,
        power=1,
    )
    logmel = librosa.amplitude_to_db(mel, top_db=top_db)
    logmel = logmel / top_db + 1
    np.save(out_path, logmel)
    # return mel, logmel
    return out_path, logmel.shape[-1]


def save_rep(utt_id_fn, save_dir, data_split=None):
    """
    utt_id_fn: identifier for utterances
    save_dir: directory where the representations are saved
    data_split: dataset split (if applicable)
    """
    start = time.time()
    os.makedirs(save_dir, exist_ok=True)
    utt_id_lst = read_lst(utt_id_fn)

    executor = ProcessPoolExecutor(max_workers=cpu_count())
    futures = []
    for item in utt_id_lst:
        utt_id, wav_path = item.split("\t")
        out_path = os.path.join(save_dir, utt_id + ".npy")
        futures.append(executor.submit(partial(process_wav, wav_path, out_path)))

    results = [future.result() for future in tqdm(futures)]

    lengths = [x[-1] for x in results]
    frames = sum(lengths)
    frame_shift_ms = 160 / 16000
    hours = frames * frame_shift_ms / 3600
    print(
        "Wrote {} utterances, {} frames ({:.2f} hours)".format(
            len(lengths), frames, hours
        )
    )
    print(f"Time for extracting filterbanks for 500 utterances: {format_time(start)}")


if __name__ == "__main__":
    fire.Fire()
