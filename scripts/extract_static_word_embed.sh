function=$1
model_name=$2
ckpt_dir=$3
subfname=$4
save_dir_pth=$5
subset_id=$6

model_type="pretrained"

if [[ $model_name == "avhubert"* ]]; then
	pckg_dir="packages/av_hubert/avhubert"
elif [[ $model_name == "wavlm"* ]]; then
	pckg_dir="packages/unilm/wavlm"
elif [[ $model_name == "fastvgs"* ]]; then
	pckg_dir="packages/FaST-VGS-Family"
else
	pckg_dir="packages/"
fi

if [[ $model_name == "fastvgs"* ]]; then
	ckpt_pth="${ckpt_dir}/${model_name}"
else
	ckpt_pth="${ckpt_dir}/${model_name}.pt"
fi
dataset="librispeech"

utt_id_fn=data_samples/librispeech/word_level/all_words/${subfname}/word_segments_${subset_id}.lst
save_dir=$save_dir_pth/${model_name}/${dataset}_word-embeddings/${subfname}

if [[ $function == "extract" ]]; then
	echo -e "\n\nExtracting ${subfname} word embeddings from ${model_name} model"
	python codes/prepare/extract_static_word_embed.py extract \
	--model_name $model_name \
	--ckpt_pth `realpath $ckpt_pth` \
	--save_dir $save_dir \
	--utt_id_fn `realpath $utt_id_fn` \
	--model_type $model_type \
	--pckg_dir `realpath $pckg_dir`
elif [[ $function == "combine" ]]; then
	echo -e "\n\nCombining ${subfname} word embeddings from ${model_name} model"
	embedding_dir=$save_dir_pth/$model_name/librispeech_word_embeddings/$subfname
	python codes/prepare/extract_static_word_embed.py combine $embedding_dir
fi
