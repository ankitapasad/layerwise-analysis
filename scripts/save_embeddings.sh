save_dir_pth=$1
alignment_dir=$2
embed_type=$3

embedding_dir=$save_dir_pth/embeddings
mkdir -p $embedding_dir

if [ "$embed_type" = "glove" ]; then
    echo "Downloading glove embeddings"
    glove_fname=glove.840B.300d
    wget -P $embedding_dir https://nlp.stanford.edu/data/${glove_fname}.zip 
    unzip $embedding_dir/${glove_fname}.zip -d $embedding_dir
    in_fn=`realpath $embedding_dir/$glove_fname.txt`
    out_fn=`realpath $embedding_dir/glove_embed.pkl`

    echo "Save as dictionary map"
    python codes/tools/save_embeddings.py save_as_dct $in_fn $out_fn
    rm $embedding_dir/${glove_fname}.zip
    rm $embedding_dir/${glove_fname}.txt
elif [ "$embed_type" = "agwe" ]; then
    echo "Downloading agwe embeddings"
    agwe_fname=librispeech_agwe_map
    wget -P $embedding_dir https://dl.ttic.edu/${agwe_fname}.zip 
    unzip -qq $embedding_dir/${agwe_fname}.zip -d $embedding_dir
    in_fn=`realpath $embedding_dir/$agwe_fname.txt`
    out_fn=`realpath $embedding_dir/agwe_embed.pkl`

    echo "Save as dictionary map"
    python codes/tools/save_embeddings.py save_as_dct $in_fn $out_fn
    rm $embedding_dir/${agwe_fname}.zip 
    rm $embedding_dir/${agwe_fname}.txt
elif [ "$embed_type" = "one-hot" ]; then
    echo -e "\n\nSaving one-hot embeddings for phone and word segments"
    python codes/prepare/read_librispeech_alignments.py one_hot word `realpath $alignment_dir` `realpath $embedding_dir` 500
    python codes/prepare/read_librispeech_alignments.py one_hot phone `realpath $alignment_dir` `realpath $embedding_dir` -1
fi
