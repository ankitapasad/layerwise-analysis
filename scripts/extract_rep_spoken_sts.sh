model_name=$1
ckpt_dir=$2
subset_id=$3
save_dir_pth=$4
pckg_dir=$5

model_type="pre-trained"
if [[ $model_name == "avhubert"* ]]; then
	pckg_dir="$pckg_dir/av_hubert/avhubert"
elif [[ $model_name == "wavlm"* ]]; then
	pckg_dir="$pckg_dir/unilm/wavlm"
elif [[ $model_name == "fastvgs"* ]]; then
	pckg_dir="$pckg_dir/FaST-VGS-Family"
elif [[ $model_name == "xlsr"* ]]; then
	pckg_dir="$pckg_dir"
else
	pckg_dir="$pckg_dir"
fi

if [[ $model_name == "fastvgs"* ]]; then
	ckpt_pth="$ckpt_dir/${model_name}"
else
	ckpt_pth="$ckpt_dir/${model_name}.pt"
fi

utt_id_fn="data_samples/spoken_sts/utt_level/split${subset_id}.tsv"
save_dir="${save_dir_pth}/${model_name}/spoken_sts/utt_level/${subset_id}"

echo "Removing any existing features extracted for sample ${data_sample}"
if [ -f $save_dir ]; then
	rm -r $save_dir
fi

echo -e "\n\nExtracting spoken STS utterance level features from ${model_type} ${model_name} model"
echo -e "for sample set $utt_id_fn"

python codes/prepare/extract_rep.py save_rep \
--model_name $model_name \
--ckpt_pth `realpath $ckpt_pth` \
--save_dir $save_dir \
--utt_id_fn `realpath $utt_id_fn` \
--model_type $model_type \
--rep_type "contextualized" \
--span "utt" \
--offset False \
--mean_pooling True \
--pckg_dir `realpath $pckg_dir`