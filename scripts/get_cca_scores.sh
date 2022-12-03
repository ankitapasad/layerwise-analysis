model_name=$1
data_sample=$2
exp_name=$3
span=$4
save_dir_pth=$5

dataset=librispeech
dataset_split=dev-clean
base_layer=0
ft_data=960h

sub_dir_name="${dataset}_${dataset_split}_sample${data_sample}"
rep_dir="$save_dir_pth/${model_name}/${sub_dir_name}"
fbank_dir="$save_dir_pth/fbanks/${sub_dir_name}"
save_dir="logs/${dataset}_${model_name}"
mkdir -p $save_dir
embed_dir="$save_dir_pth/embeddings"
rep_dir2="${save_dir_pth}/${model_name}_${ft_data}/${sub_dir_name}"
sample_data_fn="data_samples/librispeech/${span}_level/${dataset_split}_segments_sample${data_sample}_0.json"
if [ "$exp_name" = "cca_intra" ]; then
    save_fn="$save_dir/cca-wrt-layer${base_layer}_${dataset_split}_sample${data_sample}.json"
else
    save_fn="$save_dir/${exp_name}_${dataset_split}_sample${data_sample}.json"
fi

python codes/tools/get_scores.py cca \
--save_fn `realpath $save_fn` \
--rep_dir `realpath $rep_dir` \
--fbank_dir `realpath $fbank_dir` \
--exp_name $exp_name \
--base_layer $base_layer \
--model_name $model_name \
--rep_dir2 $rep_dir2 \
--embed_dir $embed_dir \
--sample_data_fn `realpath $sample_data_fn` \
--span $span
