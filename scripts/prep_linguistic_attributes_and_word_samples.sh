save_dir_pth=$1
alignment_data_dir=$2

mkdir -p data && cd data
if [ ! -d "qvec/" ]; then
    git clone https://github.com/ytsvetko/qvec.git
fi
cd ../
embedding_dir=$save_dir_pth/embeddings
mkdir -p $embedding_dir

echo "Reformat linguistic features"
python3 codes/prepare/prep_linguistic_attributes.py features "data/qvec/oracles/ptb.pos_tags" "syntactic" $embedding_dir 
python3 codes/prepare/prep_linguistic_attributes.py features "data/qvec/oracles/semcor_noun_verb.supersenses.en" "semantic" $embedding_dir 


echo "Prepare a list of words to be studied"
python3 codes/prepare/prep_linguistic_attributes.py data $embedding_dir $alignment_data_dir


echo "Prepare corresponding word segments for study"
num_instances=200
dur_thresh=10000

word_lst_pth="data_samples/librispeech/word_level/all_words.lst"
save_dir="data_samples/librispeech/word_level/all_words_${num_instances}instances"
mkdir -p $save_dir

python codes/prepare/create_data_samples.py all-words $alignment_data_dir $word_lst_pth $save_dir $dur_thresh $num_instances