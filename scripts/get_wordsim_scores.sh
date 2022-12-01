model_name=$1
subfname=$2
save_dir_pth=$3

dataset=librispeech
embedding_dir="$save_dir_pth/$model_name/librispeech_word_embeddings/$subfname"
save_dir="logs/${dataset}_${model_name}"
mkdir -p $save_dir
save_fn="${save_dir}/wordsim_$subfname.pkl"

wordsim_task_fn="tasks/wordsim/wordsim_tasks_thresh0.pkl"

python codes/tools/get_scores.py wordsim \
--save_fn `realpath $save_fn` \
--embedding_dir `realpath $embedding_dir` \
--wordsim_task_fn `realpath $wordsim_task_fn` \
--model_name $model_name \
