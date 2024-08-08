model_name=$1
save_dir_pth=$2
sts_data_dir=$3

gt_score_fn="${sts_data_dir}/spoken_sts/all_gt.json"
pair_idx_fn="${sts_data_dir}/spoken_sts/all_pair_idx.json"
res_dir="logs/"
sample_data_fn="data_samples/spoken_sts/utt_level"

python codes/tools/get_scores.py spoken_sts $model_name $save_dir_pth $gt_score_fn $pair_idx_fn $res_dir $sample_data_fn