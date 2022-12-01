model_name=$1
data_sample=$2
data_dir=$3
alignment_dir=$4
dataset_split=$5
span=$6
rep_dir_pth=$7

# data_dir="/share/data/speech/Datasets/LibriSpeech/LibriSpeech"
num_samples=500
dataset=librispeech

audio_ext=flac
dir_depth=3

if [ "$span" = "frame" ]; then	
	echo -e "\nSampling ${num_samples} utterances for extracting frame-level features"
	save_dir="data_samples/${dataset}/frame_level"
	mkdir -p $save_dir
	save_fn="${save_dir}/${num_samples}_ids_sample${data_sample}_${dataset_split}.tsv"
	python codes/prepare/create_data_samples.py frame-level $data_dir/${dataset_split} $save_fn $audio_ext $dir_depth $num_samples

	echo "Removing any existing features extracted for sample ${data_sample} as the sample set has now changed"

	mel_feat_dir=${rep_dir_pth}/fbanks/${dataset}_${dataset_split}_sample${data_sample}
	local_feat_dir=${rep_dir_pth}/${model_name}/${dataset}_${dataset_split}_sample${data_sample}/local/frame_level
	contextualized_feat_dir=${rep_dir_pth}/${model_name}/${dataset}_${dataset_split}_sample${data_sample}/contextualized/frame_level
	if [ -f $mel_feat_dir ]; then
		rm $mel_feat_dir
	fi
	if [ -f $local_feat_dir ]; then
		rm $local_feat_dir
	fi
	if [ -f $contextualized_feat_dir ]; then
		rm $contextualized_feat_dir
	fi
elif [ "$span" = "phone" ] || [ "$span" = "word" ]; then	
	echo -e "\n\nSampling ${span} segments"
	save_dir="data_samples/${dataset}/${span}_level"
	if [ "$span" = "phone" ]; then
		num_tokens=39
	elif [ "$span" = "word" ]; then
		num_tokens=500
	fi
	python codes/prepare/create_data_samples.py token-level $span $alignment_dir $dataset_split $num_tokens $data_sample $save_dir

	echo "Removing any existing features extracted for sample ${data_sample} as the sample set has now changed"

	feat_dir=${rep_dir_pth}/${model_name}/${dataset}_${dataset_split}_sample${data_sample}/contextualized/${span}_level
	if [ -f $feat_dir ]; then
		rm $feat_dir
	fi
fi
