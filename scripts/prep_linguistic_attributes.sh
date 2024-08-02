mkdir -p data && cd data
if [ ! -d "qvec/" ]; then
    git clone https://github.com/ytsvetko/qvec.git
fi
cd ../
embedding_dir=$save_dir_pth/embeddings
mkdir -p $embedding_dir
python codes/prepare/prep_linguistic_attributes.py data/qvec/oracles/ptb.pos_tags syntactic $embedding_dir 
python codes/prepare/prep_linguistic_attributes.py data/qvec/oracles/semcor_noun_verb.supersenses.en semantic $embedding_dir 