import fire
import glob
import numpy as np
import os
import textgrids
from tqdm import tqdm

curr_dir = os.path.dirname(os.path.realpath(__file__))
import sys

sys.path.insert(0, os.path.join(curr_dir, ".."))
from utils import save_dct, write_to_file, load_dct, read_lst


class LibrispeechAlign:
    def save_data(self, data_dir, dataset_split, audio_dir, audio_ext):
        """
        Save alignment info as a dictionary of token mapped to a list of occurences with time stamps
        Also, updates the count dictionary and list of tokens
        """
        self.audio_dir = audio_dir
        self.audio_ext = audio_ext
        self.dataset_split = dataset_split
        self.data_dir = data_dir

        token_lst_dct = self.read_data()
        self.get_token_alignment_ordered_lst(token_lst_dct, data_dir, dataset_split)
        token_alignment_dct = self.get_token_alignment_dct(token_lst_dct)
        for key, value in token_alignment_dct.items():
            save_dct(
                os.path.join(data_dir, f"alignment_{key}_{dataset_split}.json"), value
            )

        if "train" in dataset_split:
            count_fn, token_lst_fn = {}, {}
            for key in ["phone", "word"]:
                count_fn[key] = os.path.join(data_dir, f"{key}_count.json")
                token_lst_fn[key] = os.path.join(data_dir, f"{key}.lst")
            self.update_tokens(count_fn, token_lst_fn, token_alignment_dct)

    def read_data(self):
        """
        Read data from textgrids into a list of tuples
        """
        wrd_lst, phn_lst = [], []
        parent_dir = os.path.join(self.data_dir, self.dataset_split)
        all_fns = glob.glob(os.path.join(parent_dir, "*/*/*.TextGrid"))
        for fname in tqdm(all_fns):
            self.get_info(fname, phn_lst, wrd_lst)
        token_lst_dct = {"phone": phn_lst, "word": wrd_lst}

        return token_lst_dct

    def get_token_alignment_dct(self, token_lst_dct):
        """
        Convert a list of token-level alignments to a dictionary for a list of occurences of each token type
        """
        token_alignment_dct = {}
        for key, value in token_lst_dct.items():
            token_alignment_dct[key] = {}
            for item in tqdm(value):
                utt_id, start, end, token = item.split(" ")
                audio_path = os.path.join(
                    self.audio_dir,
                    "/".join(utt_id.split("-")[:2]),
                    utt_id + "." + self.audio_ext,
                )
                _ = token_alignment_dct[key].setdefault(token, [])
                token_alignment_dct[key][token].append((utt_id, audio_path, start, end))

        return token_alignment_dct

    def get_token_alignment_ordered_lst(self, token_lst_dct, data_dir, dataset_split):
        """
        Save the list of token-level alignments to a tsv file
        """
        for key, value in token_lst_dct.items():
            write_str = []
            for item in tqdm(value):
                write_str.append("\t".join(item.split(" ")))
            write_fn = os.path.join(data_dir, f"alignment_{key}_{dataset_split}.tsv")
            write_to_file("\n".join(write_str), write_fn)

    def phn_map(self, phn_label):
        if phn_label == "sil":
            return "SIL"
        elif phn_label[-1] in ["0", "1", "2"]:
            return phn_label[:-1].lower()
        else:
            return phn_label.lower()

    def txt_from_tier(self, tier_content, data_lst, fname, unit):
        """
        Save as filename start end label
        """
        for item in tier_content:
            label = item.text
            if label:  # check that it is non-empty
                start = str(item.xmin)
                end = str(item.xmax)
                if "phone" in unit:  # map to the traditional 39 phone phn set
                    label = self.phn_map(label)
                text_out = " ".join([fname, start, end, label])
                if label not in ["spn", "sp"]:
                    data_lst.append(text_out)

    def get_info(self, fname, phn_lst, wrd_lst):
        grid = textgrids.TextGrid(fname)
        fname = fname.split("/")[-1].split(".")[0]
        self.txt_from_tier(grid["phones"], phn_lst, fname, "phone")
        self.txt_from_tier(grid["words"], wrd_lst, fname, "word")

    def update_tokens(self, count_fn, token_lst_fn, token_alignment_dct):
        count_dct = {}
        for token_type, value in count_fn.items():
            if os.path.exists(value):
                count_dct[token_type] = load_dct(value)
            else:
                count_dct[token_type] = {}
            alignment_info_dct = token_alignment_dct[token_type]
            for token, alignment_info_lst in alignment_info_dct.items():
                _ = count_dct[token_type].setdefault(token, 0)
                count_dct[token_type][token] += len(alignment_info_lst)

            save_dct(value, count_dct[token_type])
            dct = count_dct[token_type]
            sorted_token_lst = sorted(dct, key=dct.get, reverse=True)
            write_to_file("\n".join(sorted_token_lst), token_lst_fn[token_type])


def combine_alignments(data_dir, data_split, token_type):
    combined_dct = {}
    if data_split == "train-clean":
        constitutes = ["train-clean-100", "train-clean-360"]
    elif data_split == "train":
        constitutes = ["train-clean-100", "train-clean-360", "train-other-500"]
    for sub_data_split in constitutes:
        alignment_dct = load_dct(
            os.path.join(data_dir, f"alignment_{token_type}_{sub_data_split}.json")
        )
        for token in tqdm(alignment_dct):
            alignment_lst = alignment_dct[token]
            _ = combined_dct.setdefault(token, [])
            combined_dct[token].extend(alignment_lst)
    
    save_dct(
        os.path.join(data_dir, f"alignment_{token_type}_{data_split}.json"),
        combined_dct,
    )


def save_one_hot_encodings(token, data_dir, save_dir, num_tokens=-1):
    token_lst = read_lst(os.path.join(data_dir, f"{token}.lst"))
    if token == "word":
        assert num_tokens != -1
        token_lst.remove("<unk>")
        token_lst = token_lst[:num_tokens]
    rep_mat = np.eye(len(token_lst))
    rep_dct = {token: one_hot_arr for token, one_hot_arr in zip(token_lst, rep_mat)}
    save_dct(os.path.join(save_dir, f"{token}_embed.pkl"), rep_dct)


if __name__ == "__main__":
    fire.Fire(
        {
            "read": LibrispeechAlign,
            "combine": combine_alignments,
            "one_hot": save_one_hot_encodings,
        }
    )
