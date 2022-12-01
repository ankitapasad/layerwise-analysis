"""
Implements scoring for CCA and MI

CCA: 
1. Each input data matrix is split into 10 parts
2. For <num_trials> times:
    - 8 randomly chosen parts are used for training 
    - For <num_reg_param_values> times:
        - Two random epsilon values are chosen for view1 and view2
        - The trained parameters are saved
    - Results are evaluated on the dev set for each of the <num_reg_param_values> parameters
    - Best reg_param value pair is chosen based on dev set result
    - The score is reported in test set
3. The mean of <num_trails> test set scores is saved as the CCA score    

MI: 
1. k-means clustering is performed on <all_rep> (training set)
2. Predictions of the above model on the dev set are saved
"""
from glob import glob
import numpy as np
import os
import random
import time

from scipy.spatial.distance import cosine
from scipy.stats import spearmanr
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import mutual_info_score

curr_dir = os.path.dirname(os.path.realpath(__file__))
import sys

sys.path.insert(0, os.path.join(curr_dir, ".."))
import utils
from cca_core import CCA


def logger(write_str, log_fn):
    utils.add_to_file(write_str + "\n", log_fn)


class PrepForCCA:
    def __init__(
        self,
        num_samples,
        rep_dir,
        layer_num,
        exp_name,
        label_lst=None,
        num_splits=10,
        subset=None,  # used only for cca-mel experiments
    ):
        self.num_samples = num_samples
        self.num_splits = num_splits
        self.label_lst = label_lst
        # directory to save dataset splits used for cross validation
        self.data_dir = os.path.join(
            rep_dir, "cca_specifics", exp_name, "sample_splits"
        )
        if subset is not None:
            self.data_dir = os.path.join(self.data_dir, subset)
        # directory to save learned CCA projection matrices
        self.mat_dir = os.path.join(
            rep_dir, "cca_specifics", exp_name, f"layer{layer_num}"
        )
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.mat_dir, exist_ok=True)
        self.log_fn = os.path.join(
            rep_dir, "cca_specifics", exp_name, f"cca_run_logs_layer{layer_num}"
        )
        logger("".join(["-"] * 50), self.log_fn)
        print(f"Logging at {self.log_fn}")

    def save_indices(self, idx_lst, split_num):
        """
        save sampled indices to a file
        """
        idx_lst_str = list(map(str, idx_lst))
        utils.write_to_file(
            "\n".join(idx_lst_str), os.path.join(self.data_dir, f"split{split_num}.lst")
        )

    def save_splits(self, idx_dct):
        for split_num in range(self.num_splits):
            self.save_indices(idx_dct[split_num], split_num)

    def chunk(self, indices, idx_dct):
        split_size = len(indices) // self.num_splits
        num_splits = min([self.num_splits, len(indices)])
        for split_num in range(num_splits):
            _ = idx_dct.setdefault(split_num, [])
            chosen_indices = random.sample(indices, split_size)
            idx_dct[split_num].extend(chosen_indices)
            indices = list(set(indices) - set(chosen_indices))
        assert len(indices) < self.num_splits
        for idx in indices:
            split_num = random.randrange(self.num_splits)
            idx_dct[split_num].extend([idx])

    def split_and_save_indices(self):
        """
        Split the index list into num_splits splits
        """
        indices = list(np.arange(self.num_samples))
        idx_dct, label_dct = {}, {}
        if self.label_lst is not None:
            for idx, label in enumerate(self.label_lst):
                _ = label_dct.setdefault(label, [])
                label_dct[label].append(idx)
        else:
            label_dct["all"] = indices
        for _, idx_lst in label_dct.items():
            self.chunk(idx_lst, idx_dct)
        self.save_splits(idx_dct)

    def check_splits_exist(self, expected_num_matrices, force_train):
        """
        Check whether the splits and the matrices exist, if not create the splits
        """
        num_files = len(glob(os.path.join(self.data_dir, "*.lst")))
        num_matrices = len(glob(os.path.join(self.mat_dir, "proj_*.npy")))
        if num_files != self.num_splits:
            self.split_and_save_indices()
            logger(f"{self.num_splits} file splits created", self.log_fn)
            train = True
        elif force_train:
            train = True
        else:
            logger("File splits exist", self.log_fn)
            if num_matrices != expected_num_matrices:
                train = True
            else:
                logger("Training already performed", self.log_fn)
                train = False
        if train and num_matrices > 0:
            logger("Removing existing projection matrices", self.log_fn)
            for f in glob(os.path.join(self.mat_dir, "*.npy")):
                os.remove(f)

        return train, self.data_dir, self.mat_dir, self.log_fn


class CCACrossVal:
    def __init__(
        self,
        view1,
        view2,
        num_trials,
        mat_dir,
        data_dir,
        log_fn,
        layer_num,
        num_reg_param_values,
        num_splits=10,
        mean_score=False,
        train=False,
    ):
        self.num_trials = num_trials
        self.mat_dir = mat_dir
        self.data_dir = data_dir
        self.log_fn = log_fn
        self.num_splits = num_splits
        self.view1_full = view1
        self.view2_full = view2
        self.idx_lst = self.read_all_indices()
        self.num_reg_param_values = num_reg_param_values
        self.layer_num = layer_num
        self.all_train_scores = []
        self.mean_score = mean_score
        self.train = train
        self.all_train_scores_dct = {}

    def read_all_indices(self):
        idx_lst = []
        for split_num in range(self.num_splits):
            idx_lst.append(
                list(
                    map(
                        int,
                        utils.read_lst(
                            os.path.join(self.data_dir, f"split{split_num}.lst")
                        ),
                    )
                )
            )
        return idx_lst

    def save_trained_parameters(
        self,
        proj_mat_x,
        proj_mat_y,
        x_idxs,
        y_idxs,
        mat_fn,
        score,
        epsilon_x,
        epsilon_y,
    ):
        """
        Save projection matrices, indices and score on train
        """
        np.save(mat_fn, proj_mat_x)
        np.save(mat_fn.replace("_x", "_y"), proj_mat_y)
        np.save(mat_fn.replace("proj_x", "x_idxs"), x_idxs)
        np.save(mat_fn.replace("proj_x", "y_idxs"), y_idxs)
        write_str = ", ".join(mat_fn.split("_")[2:6])
        utils.add_to_file(
            f"{self.layer_num}, {epsilon_x}, {epsilon_y}, {score}\n",
            os.path.join(
                self.mat_dir,
                "..",
                f"train_scores_all.lst",
            ),
        )

    def get_epsilon_lst(self):
        epsilon_lst = [1e-10, 1e-11, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5]
        epsilon_tuple_lst, epsilon_tuple_lst_rem = [], []
        for i in range(3):
            for j in range(3):
                epsilon_tuple_lst_rem.append((epsilon_lst[i], epsilon_lst[j]))
        np.random.shuffle(epsilon_tuple_lst_rem)
        epsilon_tuple_lst.extend(epsilon_tuple_lst_rem)
        epsilon_tuple_lst_rem = []
        for i in range(3, len(epsilon_lst)):
            for j in range(3, len(epsilon_lst)):
                epsilon_tuple_lst_rem.append((epsilon_lst[i], epsilon_lst[j]))
        np.random.shuffle(epsilon_tuple_lst_rem)
        epsilon_tuple_lst.extend(epsilon_tuple_lst_rem)

        return epsilon_tuple_lst

    def train_cca(self, dev_idx, test_idx):
        """
        train cca for num_trials trials with different epsilon values
        """
        view1, view2 = self.prepare_train_data(dev_idx, test_idx)
        cca_obj = CCA(view1, view2)
        num_successful_runs = 0
        epsilon_tuple_lst = self.get_epsilon_lst()
        exec_epsilon_lst = []
        error_epsilon_lst = []
        max_train_score = -1
        for epsilon_x, epsilon_y in epsilon_tuple_lst:
            try:
                score, params_lst = cca_obj.get_cca_score(
                    True, epsilon_x, epsilon_y, mean_score=self.mean_score
                )
                proj_mat_x, proj_mat_y, x_idxs, y_idxs = params_lst
                exec_epsilon_lst.append((epsilon_x, epsilon_y))
                if score > max_train_score:
                    max_train_score = score
                    max_epsilon_tuple = (epsilon_x, epsilon_y)
                num_successful_runs += 1
            except np.linalg.LinAlgError:
                print("Error captured, trying other parameters")
                error_epsilon_lst.append((epsilon_x, epsilon_y))
                continue
            mat_fn = os.path.join(
                self.mat_dir,
                f"proj_x_{str(epsilon_x)}_{str(epsilon_y)}_{dev_idx}_{test_idx}.npy",
            )
            self.save_trained_parameters(
                proj_mat_x,
                proj_mat_y,
                x_idxs,
                y_idxs,
                mat_fn,
                score,
                epsilon_x,
                epsilon_y,
            )
            self.all_train_scores_dct[(epsilon_x, epsilon_y, dev_idx, test_idx)] = score
            if num_successful_runs == self.num_reg_param_values:
                break
        if num_successful_runs == 0:
            logger(
                "convergence error in all runs, trying a different split", self.log_fn
            )
            return -1
        if num_successful_runs != self.num_reg_param_values:
            logger(
                f"Only {num_successful_runs} of {self.num_reg_param_values} parameters successfully trained"
            )
        self.all_train_scores.append(max_train_score)
        self.best_train_epsilon_tuples.append(max_epsilon_tuple)
        return 1

    def load_trained_parameters(self, mat_fn):
        """
        Load trained projection matrices and valid indices
        """
        proj_mat_x = np.load(mat_fn)
        proj_mat_y = np.load(mat_fn.replace("_x_", "_y_"))
        x_idxs = np.load(mat_fn.replace("proj_x", "x_idxs"))
        y_idxs = np.load(mat_fn.replace("proj_x", "y_idxs"))

        return proj_mat_x, proj_mat_y, x_idxs, y_idxs

    def score_on_held_out(self, cca_obj, mat_fn):
        proj_mat_x, proj_mat_y, x_idxs, y_idxs = self.load_trained_parameters(mat_fn)
        score, _ = cca_obj.get_cca_score(
            train=False,
            proj_mat_x=proj_mat_x,
            proj_mat_y=proj_mat_y,
            x_idxs=x_idxs,
            y_idxs=y_idxs,
            mean_score=self.mean_score,
        )
        return score

    def evaluate_cca(self):
        mat_fn_lst = glob(os.path.join(self.mat_dir, f"proj_x_*_*_*_*.npy"))
        dev_test_indices = []
        for fname in mat_fn_lst:
            _, _, _, _, dev_idx, test_idx = (
                fname.split("/")[-1].split(".")[0].split("_")
            )
            dev_test_idx = "_".join([dev_idx, test_idx])
            dev_test_indices.append(dev_test_idx)
        dev_test_indices = list(set(dev_test_indices))
        dev_indices = [int(item.split("_")[0]) for item in dev_test_indices]
        test_indices = [int(item.split("_")[1]) for item in dev_test_indices]
        assert (
            len(dev_indices) == self.num_trials
        ), "unexpected number of learned matrices, recheck training"

        score_lst = []
        for dev_idx, test_idx in zip(dev_indices, test_indices):
            epsilon_x, epsilon_y = self.select_epsilon(dev_idx, test_idx)
            mat_fn = os.path.join(
                self.mat_dir,
                f"proj_x_{str(epsilon_x)}_{str(epsilon_y)}_{dev_idx}_{test_idx}.npy",
            )
            cca_obj = CCA(
                self.view1_full.T[np.array(self.idx_lst[test_idx])].T,
                self.view2_full.T[np.array(self.idx_lst[test_idx])].T,
            )
            score = self.score_on_held_out(cca_obj, mat_fn)
            logger(f"CCA score: {score}", self.log_fn)
            score_lst.append(score)
        return np.mean(score_lst)

    def select_epsilon(self, dev_idx, test_idx):
        """
        Select the epsilon that gives the best result on development set
        """
        cca_obj = CCA(
            self.view1_full.T[np.array(self.idx_lst[dev_idx])].T,
            self.view2_full.T[np.array(self.idx_lst[dev_idx])].T,
        )
        mat_fn_lst = glob(
            os.path.join(self.mat_dir, f"proj_x_*_*_{dev_idx}_{test_idx}.npy")
        )
        score_lst = []
        epsilon_lst = []
        for mat_fn in mat_fn_lst:
            score = self.score_on_held_out(cca_obj, mat_fn)
            score_lst.append(score)
            epsilon_str = mat_fn.split("/")[-1].split("_")[2:4]
            epsilon_lst.append(list(map(float, epsilon_str)))
        best_epsilon_tuple = epsilon_lst[np.argmax(np.array(score_lst))]
        best_dev_score = np.max(score_lst)
        logger(
            f"Best epsilon: {best_epsilon_tuple}, best_score: {best_dev_score}",
            self.log_fn,
        )
        if self.train:
            corresponding_train_score = self.all_train_scores_dct[
                (best_epsilon_tuple[0], best_epsilon_tuple[1], dev_idx, test_idx)
            ]
            logger(
                f"{self.layer_num},{best_epsilon_tuple[0]},{best_epsilon_tuple[1]},{best_dev_score},{corresponding_train_score}",
                self.log_fn.replace(f"logs_layer{self.layer_num}", "interim_results"),
            )
        return best_epsilon_tuple

    def prepare_train_data(self, dev_idx, test_idx):
        train_idx_lst = []
        for split_num in range(self.num_splits):
            if split_num not in [dev_idx, test_idx]:
                train_idx_lst.extend(self.idx_lst[split_num])
        train_idx_lst = np.array(train_idx_lst)
        view1 = self.view1_full.T[train_idx_lst].T
        view2 = self.view2_full.T[train_idx_lst].T
        return view1, view2

    def save_train_score(self):
        utils.add_to_file(
            f"{self.layer_num}, {np.mean(self.all_train_scores)}\n",
            os.path.join(
                self.mat_dir,
                "..",
                f"train_scores_mean.lst",
            ),
        )

    def cca_trainer(self):
        split_lst = list(np.arange(self.num_splits))
        dev_indices, test_indices = [], []
        score_lst = []
        num_successful_runs = 0
        for _ in range(len(split_lst) // 2):
            dev_test_idx = random.sample(split_lst, 2)
            dev_idx = dev_test_idx[0]
            test_idx = dev_test_idx[1]
            split_lst = list(set(split_lst) - set(dev_test_idx))

            logger(f"\n\nDev idx: {dev_idx}, test_idx: {test_idx}", self.log_fn)
            logger("\nTraining ...", self.log_fn)
            flag = self.train_cca(dev_idx, test_idx)
            if flag == 1:
                num_successful_runs += 1
            if num_successful_runs == self.num_trials:
                break
        self.save_train_score()

    def get_score(self):
        if self.train:
            self.cca_trainer()
        logger("\nEvaluating ...", self.log_fn)
        score = self.evaluate_cca()
        return score


def get_cca_score(
    view1,
    view2,
    rep_dir,
    layer_num,
    exp_name,
    label_lst=None,
    num_trials=3,
    num_reg_param_values=6,
    force_train=False,
    subset=None,
    mean_score=False,
):
    """
    Performs 80-10-10 N-fold cross validation thrice

    view1: 2d array of shape [feat1_dim, data_points]
    view2: 2d array of shape [feat2_dim, data_points]
    rep_dir: path to representations
    layer_num: layer id on which the analysis is performed
    label_lst: 1d array of length data_points, consists of token (phone or word) labels;
               used to split data into roughly equal entropy parts
    num_trials: number of N-fold cross-validation trials to run
    num_reg_param_values: number of regularization parameters to choose from when tuning
    force_train: ignore any existing saved learned parameters and train from scratch
    subset: only used for cca-mel experiments to denote the downsampled vs original data
    mean_score: False denotes the default PWCCA implementation where the view with lower
                dimension is used in weight calculation. When set to True the result is
                the mean of scores from using either directions for weight calculation.
    """
    assert view1.shape[1] == view2.shape[1], "dimensions don't match"
    num_samples = view1.shape[1]
    num_saved_matrices = num_trials * num_reg_param_values * 2

    prep_obj = PrepForCCA(
        num_samples, rep_dir, layer_num, exp_name, label_lst, subset=subset
    )
    train, interim_data_dir, mat_dir, log_fn = prep_obj.check_splits_exist(
        num_saved_matrices, force_train
    )

    cca_eval_obj = CCACrossVal(
        view1,
        view2,
        num_trials,
        mat_dir,
        interim_data_dir,
        log_fn,
        layer_num,
        num_reg_param_values,
        mean_score=mean_score,
        train=train,
    )
    score = cca_eval_obj.get_score()

    return score


def get_mi_score(
    n_clusters,
    batch_size,
    max_iter,
    dataset_split,
    all_rep,
    all_labels,
    eval_rep=None,
    eval_labels=None,
    centroid_init=None,
):
    """
    k-means clustering for intermediate representations
    """
    start = time.time()
    kmeans = MiniBatchKMeans(
        n_clusters=n_clusters, batch_size=batch_size, max_iter=max_iter
    ).fit(all_rep)

    inertia = kmeans.inertia_ / len(all_rep)
    if "train" in dataset_split:
        rep_labels = kmeans.labels_
        upper_bound = mutual_info_score(all_labels, all_labels)
        mi_score = mutual_info_score(rep_labels, all_labels)
    else:
        rep_labels = kmeans.predict(eval_rep)
        upper_bound = mutual_info_score(eval_labels, eval_labels)
        mi_score = mutual_info_score(rep_labels, eval_labels)
    mi_score /= upper_bound
    print("Clustering step finished in %s" % (utils.format_time(start)))
    return mi_score


def get_similarity_score(task_lst, embed_dct):
    """
    Return spearman's rho between human judgements and scores from the embedding map
    """
    cosine_similarities, human_judgements = [], []
    for w1, w2, score in task_lst:
        cosine_similarities.append(1 - cosine(embed_dct[w1], embed_dct[w2]))
        human_judgements.append(score)
    srho_score, _ = spearmanr(np.array(cosine_similarities), np.array(human_judgements))
    return srho_score
