model_name=$1
data_dir=$2
alignment_dir=$3
rep_dir_pth=$4
num_instances=$5 # number of instances for each word

dataset=librispeech

# extraction for words that appear in wordsim	
echo -e "\n\nSampling word segments"

word_lst_pth="tasks/wordsim/words_of_interest_thresh0.lst" 
if [ ! -f $word_lst_pth ]; then
    wordsim_dir=tasks/word-benchmarks
    python codes/prepare/prepare_wordsim_data.py $wordsim_dir $alignment_dir tasks/wordsim 
fi

dur_thresh=10000 # threshold for parallelization
save_dir="data_samples/${dataset}/word_level/all_words"
num=`ls -d */ | wc -l`
dir_name=iter${num}_${num_instances}instances
save_dir="$save_dir/$dir_name"

python codes/prepare/create_data_samples.py all-words $alignment_dir $word_lst_pth $save_dir $dur_thresh $num_instances

echo "Removing any existing features extracted for $dir_name as the sample set has now changed"
feat_dir=${rep_dir_pth}/${model_name}/librispeech_word-embeddings/$dir_name
if [ -f $feat_dir ]; then
    rm -r $feat_dir
fi
