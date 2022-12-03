path_to_librispeech_data=$1
save_dir_pth=save
alignment_data_dir=data_samples/librispeech/alignments
data_sample=1
model_name=wav2vec_small
model_type=pretrained

# setting steps to skip the steps previously done 
steps=2

# step 1: Preparing ailgnment data
if [ $steps -le 1 ]; then
    echo "going to step 1"
    . scripts/prepare_alignment_files.sh librispeech $path_to_librispeech_data $alignment_data_dir
fi

# step 2: Creating random data samples for analysis and extracting represenatations
if [ $steps -le 2 ]; then
    echo "going to step 2"

    # Extracting frame-level representations
    rep_type_arr=("local" "contextualized")
    span=frame
    dataset_split=dev-clean
    subset_id=0 # this is not used for frame-level representations
    . scripts/create_librispeech_data_samples.sh $data_sample $path_to_librispeech_data $alignment_data_dir $dataset_split $span $save_dir_pth
    for rep_type in ${rep_type_arr[*]}; do
        . scripts/extract_rep.sh $model_name $data_sample $model_type $rep_type $span $subset_id $dataset_split $save_dir_pth
    done

    # Extracting phone-level representations
    rep_type=contextualized
    span=phone
    dataset_split_arr=("dev-clean"  "train-clean")
    for dataset_split in ${dataset_split_arr[*]}; do
        echo $dataset_split
        . scripts/create_librispeech_data_samples.sh $data_sample $path_to_librispeech_data $alignment_data_dir $dataset_split $span $save_dir_pth
        num_samples=`ls data_samples/librispeech/${span}_level/${dataset_split}_segments_sample${data_sample}_*.pkl | wc -l`
        num_samples=`expr $num_samples - 1`
        for subset_id in $(seq 0 $num_samples); do
            echo $subset_id
            . scripts/extract_rep.sh $model_name $data_sample $model_type $rep_type $span $subset_id $dataset_split $save_dir_pth
        done
    done
fi

# step 3: Save on-hot embeddings to be further used in cca-phone experiment
if [ $steps -le 3 ]; then
    echo "going to step 3"
    . scripts/save_embeddings.sh $save_dir_pth $alignment_data_dir one-hot
fi

# step 4: Example experiments evaluating property content
if [ $steps -le 4 ]; then
    echo "going to step 4"
    echo -e "\n Evaluating MI between model representations and their phone labels"
    iter_num=0
    span=phone
    for layer_num in $(seq 0 12); do
        echo $layer_num
        . scripts/get_mi_scores.sh $span $layer_num $iter_num $model_name $data_sample $save_dir_pth
    done

    echo -e "\n Evaluating CCA between model representations and their mel filterbank features"
    exp_name=cca_mel
    span=frame
    . scripts/get_cca_scores.sh $model_name $data_sample $exp_name $span $save_dir_pth

    echo -e "\n Evaluating CCA between phone-level representations and the corresponding one-hot embeddings"
    exp_name=cca_phone
    span=phone
    . scripts/get_cca_scores.sh $model_name $data_sample $exp_name $span $save_dir_pth
fi
