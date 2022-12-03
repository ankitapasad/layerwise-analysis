import fire
from glob import glob
import numpy as np
import os
from pathlib import Path
import random

curr_dir = os.path.dirname(os.path.realpath(__file__))
import sys

sys.path.insert(0, os.path.join(curr_dir, ".."))
from utils import read_lst, load_dct, save_dct, write_to_file


def sample_utterances(data_dir, save_fn, audio_ext, dir_depth, num_samples):
    """
    Save utterance ids and corresponding paths to the audio in a file
    """
    search_path = "/".join(["*"] * dir_depth)
    all_files = glob(os.path.join(data_dir, search_path + "." + audio_ext))
    chosen_fnames = random.sample(all_files, num_samples)

    chosen_sent_ids = [Path(fname).name.split(".")[0] for fname in chosen_fnames]
    write_lst = [
        "\t".join([sent_id, fname])
        for sent_id, fname in zip(chosen_sent_ids, chosen_fnames)
    ]
    write_to_file("\n".join(write_lst), save_fn)


class tokenLevelSamples:
    def __init__(
        self, data_split, data_dir, data_sample, token, save_dir, dur_threshold=10000
    ):
        self.data_split = data_split
        self.data_dir = data_dir
        self.data_sample = data_sample
        self.save_dir = save_dir
        self.token = token
        self.dur_threshold = dur_threshold  # seconds

        os.makedirs(self.save_dir, exist_ok=True)

    def sample_tokens(self, token_lst, min_cnt, max_cnt, alignment_dct):
        """
        Sample alignments such that each token has a "good" representation
        """
        sampled_alignments = {}
        tot_dur = 0
        for token in token_lst:
            all_alignments = alignment_dct[token]
            num_instances = len(all_alignments)
            if "train" in self.data_split:
                min_cnt = min([min_cnt, num_instances])
                max_cnt = min([max_cnt, num_instances])
                num_samples = random.randint(min_cnt, max_cnt)
            else:
                num_samples = min([num_instances, min_cnt])
            chosen_alignments_idx = np.random.choice(
                np.arange(0, num_instances), num_samples, replace=False
            )
            chosen_alignments = [all_alignments[idx] for idx in chosen_alignments_idx]
            for sent_id, fname, start_time, end_time in chosen_alignments:
                start_time, end_time = float(start_time), float(end_time)
                _ = sampled_alignments.setdefault(sent_id, [])
                sampled_alignments[sent_id].append((fname, start_time, end_time, token))
                tot_dur += end_time - start_time
        print("Total duration of %s spans: %.2f seconds" % (self.token, tot_dur))
        self.split_into_sublists(sampled_alignments)

    def save_to_file(self, current_sample, alignment_dct):
        print(
            "Saving %dth split of %s spans for %s sample %d"
            % (current_sample, self.token, self.data_split, self.data_sample)
        )
        save_dct(
            os.path.join(
                self.save_dir,
                f"{self.data_split}_segments_sample{self.data_sample}_{current_sample}.json",
            ),
            alignment_dct,
        )

    def split_into_sublists(self, sampled_alignments):
        """
        Split sampled alignments into sublists
        """
        current_sample, current_dur = 0, 0
        alignment_dct = {}

        for sent_id, alignment_lst in sampled_alignments.items():
            for fname, start_time, end_time, token in alignment_lst:
                current_dur += end_time - start_time
                if sent_id not in alignment_dct:
                    alignment_dct[sent_id] = [fname]
                alignment_dct[sent_id].append((start_time, end_time, token))
                if current_dur > self.dur_threshold:
                    self.save_to_file(current_sample, alignment_dct)
                    current_sample += 1
                    current_dur = 0
                    alignment_dct = {}
        if current_dur > 0:
            self.save_to_file(current_sample, alignment_dct)

    def sample_phone_alignments(self, num_phones=39):
        """
        Sample phone alignments for MI experiments
        """
        if "train" in self.data_split:
            min_cnt, max_cnt = 3000, 7000
        else:
            min_cnt, max_cnt = 200, 1e6
        phn_lst = read_lst(os.path.join(self.data_dir, "phone.lst"))
        alignment_dct = load_dct(
            os.path.join(self.data_dir, f"alignment_phone_{self.data_split}.json")
        )
        phn_lst.remove("SIL")
        assert len(phn_lst) == num_phones
        self.sample_tokens(phn_lst, min_cnt, max_cnt, alignment_dct)

    def sample_word_alignments(self, num_words=500):
        """
        Sample word alignments for MI experiments
        """
        # from nltk.corpus import stopwords
        # english_stop_words = stopwords.words("english")
        if "train" in self.data_split:
            if num_words == 350:
                min_cnt = 800
            elif num_words == 500:
                min_cnt = 600
            max_cnt = 1200
        else:
            min_cnt, max_cnt = 15, 1e6
        alignment_dct = load_dct(
            os.path.join(self.data_dir, f"alignment_word_{self.data_split}.json")
        )
        wrd_lst = read_lst(os.path.join(self.data_dir, "word.lst"))
        wrd_lst.remove("<unk>")
        # wrd_lst = list(set(wrd_lst) - set(english_stop_words))[:num_words]
        self.sample_tokens(wrd_lst[:num_words], min_cnt, max_cnt, alignment_dct)


class AllWrdSegments:
    def __init__(
        self,
        alignment_data_dir,
        word_lst_pth,
        save_dir,
        dur_thresh=10000,
        num_instances=200,
    ):
        self.data_dir = alignment_data_dir
        self.dur_thresh = dur_thresh
        self.max_cnt = num_instances
        self.word_lst_pth = word_lst_pth
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

    def get_tot_dur(self, wrd_segment_lst):
        start_times = np.array([float(item[2]) for item in wrd_segment_lst])
        end_times = np.array([float(item[3]) for item in wrd_segment_lst])
        tot_num_secs = np.sum(end_times - start_times)
        return tot_num_secs

    def find_valid_split_idx(self, split_to_dur, tot_segment_dur):
        if tot_segment_dur > self.dur_thresh:
            keys = [split_idx for split_idx, dur in split_to_dur.items() if dur == 0]
        else:
            keys = [
                split_idx
                for split_idx, dur in split_to_dur.items()
                if (tot_segment_dur + dur) < self.dur_thresh
            ]
        if len(keys) == 0:
            return len(split_to_dur)
        else:
            return keys[0]

    def sample_word_segments(self):
        alignment_dct = load_dct(
            os.path.join(self.data_dir, f"alignment_word_train.json")
        )
        curr_split_idx = 0
        split_to_segments = {0: []}
        split_to_dur = {0: 0}
        split_to_labels = {}
        wrd_lst = read_lst(self.word_lst_pth)
        for wrd in wrd_lst:
            tot_num_wrd_segments = len(alignment_dct[wrd])
            num_samples = np.min([self.max_cnt, tot_num_wrd_segments])
            chosen_alignments_idx = np.random.choice(
                np.arange(0, tot_num_wrd_segments), num_samples, replace=False
            )
            wrd_segments = [alignment_dct[wrd][idx] for idx in chosen_alignments_idx]
            tot_segment_dur = self.get_tot_dur(wrd_segments)
            split_idx = self.find_valid_split_idx(split_to_dur, tot_segment_dur)
            _ = split_to_segments.setdefault(split_idx, [])
            _ = split_to_dur.setdefault(split_idx, 0)
            _ = split_to_labels.setdefault(split_idx, [])
            split_to_dur[split_idx] += tot_segment_dur
            split_to_segments[split_idx].extend(wrd_segments)
            split_to_labels[split_idx].extend([wrd] * len(wrd_segments))
        for split_idx, labels_lst in split_to_labels.items():
            segment_lst = [
                ",".join(list(item)) for item in split_to_segments[split_idx]
            ]
            write_to_file(
                "\n".join(labels_lst),
                os.path.join(self.save_dir, f"labels_{split_idx}.lst"),
            )
            write_to_file(
                "\n".join(segment_lst),
                os.path.join(self.save_dir, f"word_segments_{split_idx}.lst"),
            )


def sample_segments(
    token_type,
    data_dir,
    data_split,
    num_tokens,
    data_sample,
    save_dir,
    dur_threshold=10000,
):
    sample_obj = tokenLevelSamples(
        data_split, data_dir, data_sample, token_type, save_dir, dur_threshold
    )
    getattr(sample_obj, f"sample_{token_type}_alignments")(num_tokens)


def sample_all_word_instances(
    alignment_data_dir, word_lst_pth, save_dir, dur_thresh=10000, num_instances=200
):
    """
    Sample word instances for processing word-by-word
    """
    sample_obj = AllWrdSegments(
        alignment_data_dir, word_lst_pth, save_dir, dur_thresh, num_instances
    )
    sample_obj.sample_word_segments()


if __name__ == "__main__":
    fire.Fire(
        {
            "frame-level": sample_utterances,
            "token-level": sample_segments,
            "all-words": sample_all_word_instances,
        }
    )
