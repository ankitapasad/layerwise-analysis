span=$1
layer_id=$2
iter_num=$3
model_name=$4
data_sample=$5
save_dir_pth=$6

dataset=librispeech
eval_dataset_split=dev-clean
train_dataset_split=train-clean
rep_type=contextualized

sample_data_dir="data_samples/librispeech/${span}_level"
rep_dir="$save_dir_pth/${model_name}/${dataset}_dev-clean_sample${data_sample}/${rep_type}/${span}_level"

save_dir="logs/${dataset}_${model_name}"
mkdir -p $save_dir
save_fn="$save_dir/mi_${span}_${eval_dataset_split}_${train_dataset_split}.lst"

python codes/tools/get_scores.py mi \
--eval_dataset_split $eval_dataset_split \
--sample_data_dir `realpath $sample_data_dir` \
--rep_dir `realpath $rep_dir` \
--save_fn `realpath $save_fn` \
--layer_id $layer_id \
--span $span \
--iter_num $iter_num \
--data_sample $data_sample \
--train_dataset_split $train_dataset_split
