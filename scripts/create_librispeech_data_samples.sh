data_sample=$1
data_dir=$2
alignment_dir=$3
dataset_split=$4
span=$5

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
elif [ "$span" = "phone" ] || [ "$span" = "word" ]; then	
	echo -e "\n\nSampling ${span} segments"
	save_dir="data_samples/${dataset}/${span}_level"
	if [ "$span" = "phone" ]; then
		num_tokens=39
	elif [ "$span" = "word" ]; then
		num_tokens=500
	fi
	python codes/prepare/create_data_samples.py token-level $span $alignment_dir $dataset_split $num_tokens $data_sample $save_dir
fi
