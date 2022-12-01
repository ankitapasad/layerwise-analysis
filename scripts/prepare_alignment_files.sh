dataset=$1
data_dir=$2
alignment_data_dir=$3
# data_dir="/share/data/speech/Datasets/LibriSpeech/LibriSpeech"
audio_ext=flac

mkdir -p $alignment_data_dir
if [ "$dataset" = "librispeech" ]; then
	wget -P $alignment_data_dir https://zenodo.org/record/2619474/files/librispeech_alignments.zip
	unzip -qq $alignment_data_dir/librispeech_alignments.zip -d $alignment_data_dir
	rm $alignment_data_dir/librispeech_alignments.zip

    # for data_split in dev-clean dev-other train-clean-100 train-clean-360 train-other-500 test-clean test-other
    for data_split in dev-clean train-clean-100 train-clean-360
    do	
        echo $data_split
        audio_dir="${data_dir}/${data_split}"
        python codes/prepare/read_librispeech_alignments.py read save_data $alignment_data_dir $data_split $audio_dir $audio_ext
        rm -r $alignment_data_dir/$data_split
    done

    # for data_split in train-clean train
    for data_split in train-clean
    do
        echo $data_split
        python codes/prepare/read_librispeech_alignments.py combine $alignment_data_dir $data_split phone
        python codes/prepare/read_librispeech_alignments.py combine $alignment_data_dir $data_split word
    done
fi
