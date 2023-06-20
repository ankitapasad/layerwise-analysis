"""
Plot semantic scores in the ICASSP style format

Takes results saved as json file: example /share/data/speech/hackathon_2022/results/mean_scores/hubert_small_cca_ci_syntactic.json
"""

import fire
import numpy as np
import os
import sys

curr_dir = os.path.dirname(os.path.realpath(__file__))
import sys

sys.path.insert(0, os.path.join(curr_dir, ".."))
from utils import read_lst, load_dct, save_pkl, write_to_file

import matplotlib
matplotlib.rcParams.update({'figure.autolayout': True})
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.ioff()
plt.rc('text', usetex=True)   # Might require installing TeX fonts
plt.rc('axes', facecolor='w', labelcolor='k', edgecolor='k')

plt.rcParams['font.size'] = 10
## adjusted relative to font.size, using the following values: 
## xx-small, x-small, small, medium, large, x-large, xx-large, larger, or smaller
# plt.rc('font', size=14)          # controls default text sizes
# plt.rc('axes', titlesize=18)     # fontsize of the axes title
# plt.rc('axes', labelsize=18)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=11)    # fontsize of the tick labels
plt.rc('ytick', labelsize=11)    # fontsize of the tick labels
# plt.rc('ytick', labelsize=18)    # fontsize of the tick labels
plt.rc('legend', fontsize=14)    # legend fontsize
# plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

class PlotCCAScores():
    def __init__(self, ext="png"):
        parent_dir = "/share/data/lang/users/ankitap/ap-rep/git_check/layerwise-analysis/plots"
        self.plot_dir = os.path.join(parent_dir, "semantic_analysis", ext)
        os.makedirs(self.plot_dir, exist_ok=True)
        self.res_dir = "/share/data/speech/hackathon_2022/results"
        self.exp_name_to_y_lim = {
            # "semantic": {"small": [0.45, 0.58], "large": [0.5, 0.7]},
            # "syntactic": {"small": [0.3, 0.65], "large": [0.35, 0.7]},
            # "wordsim_relatedness": {"small": [0, 0.1], "large": [0, 0.18]},
            # "wordsim_similarity": {"small": [0, 0.15], "large": [0, 0.2]},
            # "spoken_sts": {"small": [0.2, 0.65], "large": [0, 0.6]},
            "glove": [-0.02, 0.38],
            "syntactic": [-0.02, 0.78]
            }
        self.exp_name_map = {
            "semantic": "Semantics (SemCor)",
            "glove": "Semantics (GloVe)",
            "syntactic": "Syntactic content",
            "wordsim_relatedness": "WordSim relatedness",
            "wordsim_similarity": "WordSim similarity",
            "spoken_sts": "Semantics (STS)",
            "agwe": "Acoustic word content",
            }
        self.model_names = {
            "small": ["wavlm_small", "wavlm_small_plus", "hubert_small", "wav2vec_small", "xlsr53_56k", "fastvgs_coco", "fastvgs_plus_coco", "avhubert_small_lrs3_vc2", "randominit_small"],#, "speechlm_small"],#, "fastvgs_coco"],
            "large": ["wavlm_large", "hubert_large", "wav2vec_vox", "xlsr53_56k", "avhubert_large_lrs3_vc2", "randominit_large"]#, "speechlm_large"]
        }
        self.baselines = ["glove", "agwe", "rand_word", "fbank", "naive"]
        self.model_name_map = {
            "wav2vec_small": "w2v2",
            "wav2vec_vox": "w2v2",
            "fastvgs_plus_coco": "fastvgs+",
            "fastvgs_coco": "fastvgs",
            "randominit_small": "rand-init",
            "randominit_large": "rand-init",
            "wavlm_small_plus": "wavlm_plus",
            }
        self.model_name_map_og = {
            "w2v2": "W2V2",
            "fastvgs+": "FaST-VGS+",
            "fastvgs": "FaST-VGS",
            "rand-init": "rand-init",
            "hubert": "HuBERT",
            "wavlm": "WavLM",
            "avhubert": "AV-HuBERT",
            "xlsr53": "XLSR-53",
            "wavlm_plus": "WavLM+",
            "speechlm": "SpeechLM",
            "glove": "GloVe",
            "agwe": "AGWE",
            "rand": "rand",
            "fbank": "FBank",
            # "naive": r'na$\ddot{i}$ve'
            "naive": "naive"
            }
        self.model_name_to_style = {
            "wavlm": {"clr": "k", "linestyle": "-", "marker": "."},
            "wavlm_plus": {"clr": "k", "linestyle": ":", "marker": ""},
            "hubert": {"clr": "green", "linestyle": "-", "marker": "."},
            "w2v2": {"clr": "#DC7633", "linestyle": "--", "marker": ""},
            # "hubert": {"clr": "green", "linestyle": "-", "marker": "x"},
            # "w2v2": {"clr": "#DC7633", "linestyle": "--", "marker": ","},
            "xlsr53": {"clr": "dodgerblue", "linestyle": "-", "marker": "1"},
            "fastvgs": {"clr": "#717D7E", "linestyle": "-", "marker": "|"},
            # "fastvgs+": {"clr": "#76D7C4", "linestyle": "-", "marker": "d"},
            "fastvgs+": {"clr": "#76D7C4", "linestyle": "-", "marker": "."},
            "avhubert": {"clr": "red", "linestyle": ":", "marker": ""},
            # "avhubert": {"clr": "red", "linestyle": ":", "marker": ","},
            "rand-init": {"clr": "#7D3C98", "linestyle": "-", "marker": ","},
            "rand_word": {"clr": "#7D3C98", "linestyle": "-", "marker": ","},
            "speechlm": {"clr": "blue", "linestyle": "--", "marker": ""},
            "agwe": {"clr": "dodgerblue", "linestyle": "-", "marker": ""},
            "fbank": {"clr": "dodgerblue", "linestyle": "-", "marker": ""},
            "glove": {"clr": "#da1fe0", "linestyle": "-", "marker": ""},
            "naive": {"clr": "blue", "linestyle": "-", "marker": ""},
        } 
        self.fname_map = {
            "semantic": "qvec_semantic",
            "syntactic": "qvec_syntactic",
            "wordsim_relatedness": "wordsim_iter0_200instances_split2",
            "wordsim_similarity": "wordsim_iter0_200instances_split2",
            "spoken_sts": "spoken_sts_1",
        }
        self.ext = ext

    def model_legend_label(self, model_name):
        if model_name in self.model_name_map:
            return self.model_name_map[model_name]
        else:
            return model_name.split("_")[0] 

    def read_scores(self, model_name, exp_name, model_type, my_xticks, x, baseline=False, baseline_embed=None):
        if baseline:
            assert baseline_embed is not None
            fname = os.path.join(self.res_dir, "mean_scores", f"baseline_cca_ci_{exp_name}_vs_{baseline_embed}_embed.json")
        else:
            fname = os.path.join(self.res_dir, "mean_scores", f"{model_name}_cca_ci_{exp_name}.json")
            # fname = os.path.join(self.res_dir, f"librispeech_{model_name}", f"{self.fname_map[exp_name]}.json")
        scores_exist = False
        if os.path.exists(fname):
            scores_exist = True
            try:
                res_dct = load_dct(fname)
            except:
                print("Error reading scores")
                import pdb; pdb.set_trace()
            if "wordsim" in exp_name:
                task_name = exp_name.split("_")[1]
                if baseline:
                    mean_score_lst = [res_dct[f"{task_name} tasks"]['0']]*len(my_xticks)
                else:
                    mean_score_lst = [res_dct[f"{task_name} tasks"][str(key)] for key in x]
            else:
                try:
                    if baseline:
                        mean_score_lst = [res_dct['0']]*len(my_xticks)
                    else:
                        if model_type == "small" and "fastvgs" not in model_name and "sts" in exp_name:
                            mean_score_lst = [res_dct[key] for key in my_xticks[:13]]
                        else:
                            mean_score_lst = [res_dct[key] for key in my_xticks]
                except:
                    print("Error retrieving scores")
                    import pdb;pdb.set_trace()
        else:
            return scores_exist, -1
        return scores_exist, mean_score_lst

    def plot_scores(self, exp_name, plot_type="small"):
        x_label = "Transformer layer number"
        y_label = self.exp_name_map[exp_name]
        fig, ax = plt.subplots(nrows=2, ncols=1)
        for idx, model_type in list(enumerate(self.model_names.keys())):
            single_score_exists = False
            for idx1, model_name in enumerate(self.model_names[model_type]):
                model_name_2 = self.model_legend_label(model_name)
                style_params = self.model_name_to_style[model_name_2]
                if idx1 == 0:
                    # if model_type == "small" and "sts" in exp_name:
                    #     num_layers = 15
                    # el
                    if model_type == "small":
                        num_layers = 12
                    else:
                        num_layers = 24
                    x = np.arange(num_layers+1)
                    my_xticks = [str(item) for item in list(x)]
                    # if model_type == "small" and "sts" in exp_name:
                    #     my_xticks[-1] = "cls"
                scores_exist, mean_score_lst = self.read_scores(model_name, exp_name, model_type, my_xticks, x)
                single_score_exists = single_score_exists or scores_exist
                if scores_exist:
                    ax[idx].plot(
                    x[:len(mean_score_lst)],
                    mean_score_lst,
                    linestyle=style_params["linestyle"],
                    color=style_params["clr"],
                    marker=style_params["marker"],
                    label=self.model_name_map_og[model_name_2],
                    lw=2.0
                    )
            for baseline_name in self.baselines:
                if baseline_name != exp_name:
                    model_name_2 = self.model_legend_label(baseline_name)
                    style_params = self.model_name_to_style[baseline_name]
                    scores_exist, mean_score_lst = self.read_scores(baseline_name, exp_name, model_type, my_xticks, x, baseline=True, baseline_embed=baseline_name)
                    single_score_exists = single_score_exists or scores_exist
                    if scores_exist:
                        # lower_lim, upper_lim = self.exp_name_to_y_lim[exp_name][model_type]
                        label_str = f"{self.model_name_map_og[model_name_2]}"
                        # if mean_score_lst[0] < lower_lim or mean_score_lst[0] > upper_lim:
                        #     label_str += f" ({np.round(mean_score_lst[0], 2)})"
                        ax[idx].plot(
                        x[:len(mean_score_lst)],
                        mean_score_lst,
                        linestyle=style_params["linestyle"],
                        color=style_params["clr"],
                        marker=style_params["marker"],
                        label= label_str,
                        lw=2.0
                        )
            if single_score_exists:
                if model_type == "large" and plot_type == "small":
                    ax[idx].set_xticks(x[::2])
                    ax[idx].set_xticklabels(my_xticks[::2])
                else:
                    ax[idx].set_xticks(x)
                    ax[idx].set_xticklabels(my_xticks)
                if "wordsim" in exp_name:
                    ax[idx].set_ylabel(y_label, fontsize=14)
                else:
                    ax[idx].set_ylabel(y_label, fontsize=16)
                if exp_name in self.exp_name_to_y_lim:
                    ax[idx].set_ylim(self.exp_name_to_y_lim[exp_name])
                    # ax[idx].set_ylim(self.exp_name_to_y_lim[exp_name][model_type])
                ax[idx].grid(b=True, linestyle='-', linewidth=0.5)
        if plot_type == "small":
            leg = ax[0].legend(bbox_to_anchor=(1, 0.9), loc='upper left', frameon=False, fontsize=12, ncol=2, columnspacing=1)
        ax[1].set_xlabel(x_label, fontsize=16)
        save_name = os.path.join(self.plot_dir, f"cca_ci_{exp_name}_{plot_type}.{self.ext}")
        print(save_name)
        plt.savefig(dpi=300, bbox_inches='tight', fname=save_name)
        plt.close()

def main(exp_name, plot_type, ext):
    plot_obj = PlotCCAScores(ext)
    plot_obj.plot_scores(exp_name, plot_type)

if __name__ == "__main__":
    fire.Fire(main) 
