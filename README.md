This codebase puts together tools and experiments to analyze self-supervised speech representations. These analysis techniques can be used to replicate the findings presented in 
1. [Layer-Wise analysis of a self-supervised speech representation model](https://arxiv.org/abs/2107.04734)
2. [Comparative layer-wise analysis of self-supervised speech models](https://arxiv.org/abs/2211.03929)
3. [What do self-supervised speech models know about words?](https://arxiv.org/abs/2307.00162)

<img src="https://github.com/ankitapasad/layerwise-analysis/blob/main/fig/all-in-one.jpg" data-canonical-src="https://github.com/ankitapasad/layerwise-analysis/blob/main/fig/all-in-one.jpg" width="650" height="400" />

# Table of Contents
- [Current support](#current-support)
  - [Pre-trained models](#pre-trained-models)
  - [Analysis experiments](#analysis-experiments)
- [Setup and Installation](#setup-and-installation)
- [Usage](#usage)
  - [0. Quick intro with an example script](#0-quick-intro-with-an-example-script)
  - [1. Data preparation](#1-data-preparation)
    - [a. Download LibriSpeech](#a-download-librispeech)
    - [b. Prepare LibriSpeech alignments](#b-prepare-librispeech-alignments)
    - [c. Prepare sampled data for analysis](#c-prepare-sampled-data-for-analysis)
    - [d. Prepare linguistic features and corresponding samples](#d-prepare-linguistic-features-and-corresponding-samples)
    - [e. Prepare other embedding maps](#e-prepare-other-embedding-maps)
    - [f. Prepare spoken STS data](#f-prepare-spoken-sts)
  - [2. Feature extraction](#2-feature-extraction)
  - [3. Evaluate layer-wise property trends](#3-evaluate-layer-wise-property-trends)
    - [Canonical Correlation Analysis](#1-canonical-correlation-analysis)
    - [Mutual information](#2-mutual-information)
  - [4. Evaluate training-free tasks](#4-evaluate-training-free-tasks)
    - [Textual word similarity](#1-textual-word-similarity)
      - [Extraction of context-independent word embeddings](#extraction-of-context-independent-word-embeddings)
      - [Evaluate WordSim](#1-wordsim-evaluation)
    - [Acoustic word disctrimination](#2-acoustic-word-discrimination)
    - [Unsupervised word segmentation](#3-unsupervised-word-segmentation)
      - [Librispeech](#segmentation-for-librispeech-dataset)
      - [Buckeye](#segmentation-for-buckeye-dataset)
    - [Spoken sentence similarity](#4-spoken-sentence-similarity)
  
# Current support
## Pre-trained models
The codebase currently supports data loading and feature extraction for the following publicly available _pre-trained_ models:
1. [wav2vec 2.0](https://github.com/pytorch/fairseq/blob/main/examples/wav2vec)
2. [HuBERT](https://github.com/pytorch/fairseq/tree/main/examples/hubert)
3. [XLSR and XLS-R](https://github.com/pytorch/fairseq/blob/main/examples/wav2vec)
4. [WavLM](https://github.com/microsoft/unilm/tree/master/wavlm)
5. [AV-HuBERT](https://github.com/facebookresearch/av_hubert)
6. [FaST-VGS and FaST-VGS+](https://github.com/jasonppy/FaST-VGS-Family)

## Analysis experiments
The following canonical correlation analysis (CCA) and mutual information (MI) experiments are currently supported by the codebase (please refer our papers for more details on individual experiments):
1. cca-intra
2. cca-mel
3. cca-phone
4. cca-word
5. cca-agwe
6. cca-glove
7. cca-semantics
8. cca-syntactic
9. mi-phone
10. mi-word

# Setup and Installation

## 1. Clone the repo and install requirements
```
git clone https://github.com/ankitapasad/layerwise-analysis.git
pip install -r requirements.txt

# for wordsim
git submodule update
git submodule init
```
Note that since this repo is intended to be used for one or more of the models listed above, **please make sure to install these libraries in the same environment that has all the necessary installations and dependencies for the corresponding model(s)**.

## 2. Pre-trained checkpoints and related setup
Install the relevant model packages to `$pckg_dir` and download the pre-trained models in `$ckpt_dir`. The default inference in most model packages does not directly return layerwise outputs. This can that can be easily fixed with minor edits to their model files. The edited files are added to the `modellib_addons/` directory.

**Replace the original files in the model packages with these edited versions before proceeding.**

# Usage

Follow the next steps in order to generate property content trends for pretrained models from raw wavforms and alignments. Each step is accompanied by a short explanation.

## 0. Quick intro with an example script

You can find an abridged and collated version of these steps at [examples/recipe.sh](https://github.com/ankitapasad/layerwise-analysis/blob/main/examples/recipe.sh). 

```
. examples/recipe.sh $path_to_librispeech_data $ckpt_dir $pckg_dir
``` 
Perform step 1a below, and read the accompanying [README.md](https://github.com/ankitapasad/layerwise-analysis/blob/main/examples/README.md) before running the script. 

## 1. Data preparation
### a. Download LibriSpeech 
Currently, all the experiments use [Librispeech](https://www.openslr.org/12), so before proceeding further make sure you have the dataset downloaded. Download and extract all the files into the `$path_to_librispeech_data` directory, such that this directory has a folder for each dataset split.

### b. Prepare LibriSpeech alignments
Follow the next two steps to prepare data files.

```
bash scripts/prepare_alignment_files.sh librispeech $path_to_librispeech_data $alignment_data_dir
```
This will download and reformat the phone and word alignment files for Librispeech and save the alignments as dictionary files to `$alignment_data_dir`. These `.json` files map each phone/word type to a list of tuples `(utt_id, path_to_wav, start_time, end_time)`. This might take 30 minutes. _Note_: Uncomment step #3 commands if you intend to apply MI tools, this will add a few more minutes of processing time.  

### c. Prepare sampled data for analysis
```
data_sample=1
dataset_split=dev-clean
span=frame
bash scripts/create_librispeech_data_samples.sh $data_sample $path_to_librispeech_data $alignment_data_dir $dataset_split $span
```
This will randomly sample audio utterances and phone and word segment instances from Librispeech. The list of sampled utterance ids will be saved to the `data_samples/librispeech` directory.

##### _What is `data_sample`?_
The `data_sample` variable is an identifier for the data sample set. You can have more than one data_sample sets, generated by passing a different identifier, and repeat all the analysis experiments on each data_sample set to check for robustness (using mean and standard deviation).

##### _What is `subset_id`?_
Understanding this is not necessary for most CCA experiments since those are performed on the smaller dev splits. The following is relevant when sampling and processing utterances from train splits (for CCA-semantic and CCA-syntactic experiments).

For phone and word segments, the sampled set can be split further into subsets identified with numbers `0, 1, 2, ...`. These ids are referred to as `subset_id` in the next steps. This is done so that each subset can be processed in parallel for feature extraction. The extracted representations for each subset are concatenated into one single representation matrix before evaluating analysis scores.

Currently, each subset is under 10000 seconds. This threshold can be changed by passing the `dur_threshold` argument to the [`create_data_samples.py token-level` line](https://github.com/ankitapasad/layerwise-analysis/blob/main/scripts/create_librispeech_data_samples.sh#L45). 

### d. Prepare linguistic features and corresponding samples
We generate the word samples for linguistic features separately because these experiments use a higher coverage of vocabulary.

Save formatted semantic and syntactic attributes and create the word samples for analysis.
```
bash scripts/prep_linguistic_attributes_and_word_samples.sh $save_dir_pth $alignment_data_dir
```

### e. Prepare other embedding maps
Download and store GloVe and AGWE embedding maps as dictionary files.
```
bash scripts/save_embeddings.sh $save_dir_pth $alignment_data_dir agwe
bash scripts/save_embeddings.sh $save_dir_pth $alignment_data_dir glove
bash scripts/save_embeddings.sh $save_dir_pth $alignment_data_dir one-hot
```

### f. Prepare spoken STS
Install [datasets package](https://huggingface.co/docs/datasets/en/installation) in your environment and download and prepare spoken STS data in `$path_to_sts_data`. 

```
bash scripts/prep_spoken_sts.sh $path_to_sts_data
```


## 2. Feature extraction

Example: Extract representations from the pre-trained wav2vec2.0 model for dev-clean split
```
model_name=wav2vec_base
dataset_split=dev-clean
data_sample=1
subset_id=0
```

Set `save_dir_pth` as the directory where the extracted representations will be saved. The script will generate sub-folders within `$save_dir_pth` for each different model and each different type of representation. **Pass the same `$save_dir_pth` variable to all the subsequent scripts.**

Frame-level representations from the 7 convolutional layers 

```
rep_type=local
span=frame
bash scripts/extract_rep.sh $model_name $ckpt_dir $data_sample $rep_type $span $subset_id $dataset_split $save_dir_pth $pckg_dir librispeech
```

Similarly for extracting representations from transformer layers, **run the above script with following changes to the arguments**: 
- Frame-level representations from all transformer layers: `rep_type=contextualized; span=frame`
- Mean-pooled phone-level representations from all transformer layers:`rep_type=contextualized; span=phone`
- Mean-pooled word-level representations from all transformer layers:`rep_type=contextualized; span=word`

The extracted features will be saved at `$save_dir_pth/$model_name/librispeech_$dataset_split_sample1` directory

For a larger vocabulary coverage of words (these representations are primarily used for semantic and syntactic features)
- Mean-pooled word-level representations from all transformer layers: `rep_type=contextualized; span=all_words`
- The extracted features will be saved at `$save_dir_pth/$model_name/librispeech_all_words_200instances/$subset_id`
- Once all the `subset_id`s are processed, combine the representations to form an embedding map, which will be stored at `$save_dir_pth/$model_name/librispeech_all_words_200instances`
```
python codes/prepare/extract_rep.py combine $model_name $save_dir_pth "all_words_200instances"
```

## 3. Evaluate layer-wise property trends
The results will be saved at `logs/librispeech_${model_name}/`.

### 1. Canonical correlation analysis
Example: Canonical correlation analysis between the extracted representations and mel filterbank features for representations extracted at a frame-level
```
exp_name=cca_mel
span=frame
bash scripts/get_cca_scores.sh $model_name $data_sample $exp_name $span $save_dir_pth
```

In order to evaluate a single layer at a time, pass `$layer_num` to the same script
```
exp_name=cca_mel
span=frame
layer_num=T4 # process transformer layer 4
bash scripts/get_cca_scores.sh $model_name $data_sample $exp_name $span $save_dir_pth $layer_num
```

The following CCA experiments are possible:
| $exp_name | $span |
| --------- | ----- |
| cca_mel   | frame |
| cca_intra | frame |
| cca_phone | phone |
| cca_word  | word |
| cca_agwe  | word |
| cca_glove | word |
| cca_semantic | word |
| cca_syntactic | word |

### 2. Mutual information
Example: Mutual information between $span labels and the extracted phone segments for representations extracted from layer 1. The `iter_num` denotes the iteration number. To account for the randomness introduced by k-means clustering, we run this experiment for each sample set multiple times.
```
iter_num=0
layer_num=1
span=phone # or word
bash scripts/get_mi_scores.sh $span $layer_num $iter_num $model_name $data_sample $save_dir_pth
```

## 4. Evaluate training-free tasks
### 1. Textual word similarity 
#### Extraction of context-independent word embeddings

- Generate samples of words from the train set. Note that there is a `num_instances` variable inside the script, that is the value for number of instances' representations averaged for each word embedding. 
```
bash scripts/create_wsim_word_samples.sh $model_name $path_to_librispeech_data $alignment_data_dir $save_dir_pth $num_instances
```
This will sample the word instances and divide all words into subsets, such that each subset is smaller than 10000 seconds (for processing speech and parallelization).

- Extract representation, you'll run the following for each `subset_id`. The argument `subfname` denotes the sample directory name that is set [here](https://github.com/ankitapasad/layerwise-analysis/blob/main/scripts/create_wsim_word_samples.sh#L22) and for which you wish to extract the embeddings.
```
bash scripts/extract_static_word_embed.sh extract $model_name $ckpt_dir $subfname $save_dir_pth $subset_id
```

- Once all the subsets are processed, combine the representations to form an embedding map.
```
bash scripts/extract_static_word_embed.sh combine $model_name $ckpt_dir $subfname $save_dir_pth
```

#### Evaluate WordSim
```
bash scripts/get_wordsim_scores.sh $model_name $subfname $save_dir_pth
```

### 2. Acoustic word discrimination
[Coming soon]

### 3. Unsupervised word segmentation
#### Segmentation for LibriSpeech dataset
Run the script below to perform word segmentation on the LibriSpeech dataset:
```
python3 codes/tools/word_segmentation_librispeech.py $save_dir_pth/$model_name/librispeech_$dataset_split_sample1/contextualized/frame_level/ data_samples/librispeech/frame_level/500_ids_sample1_dev-clean.tsv $path_to_librispeech_data $librispeech_alignment_data_dir
```
It automatically conducts grid search to find the best combination of hyper-parameters based on the F-scores computed on detected word boundaries.

#### Segmentation for Buckeye dataset
Follow [Herman's repository](https://github.com/kamperh/vqwordseg?tab=readme-ov-file) to prepare the data and ground-truth word boundaries for the Buckeye dataset. 
Then, Create a tsv file in a similar format as `data_samples/librispeech/frame_level/500_ids_sample1_dev-clean.tsv`, which includes the file ID and path to the audio file of each sentence in the dataset.
An example can be found in `example_files/data_samples/buckeye/segmentation/buckeye_val.tsv`.
Extract frame-level representations in a similar way as [step 2](https://github.com/ankitapasad/layerwise-analysis/tree/main?tab=readme-ov-file#2-feature-extraction).
```
dataset_split={val or test}
. ./scripts/extract_rep.sh $model_name $ckpt_dir $data_sample contextualized frame 1 $dataset_split $save_dir_pth $pckg_dir buckeye
```

Run the script below to predict word boundaries for the Buckeye dataset. The optimal hyper-parameters found in the previous step with the LibriSpeech dataset can be used.
```
prominence=$optimal_prominence
dist=$optimal_dist
window_size=$optimal_window_size

python3 codes/segmentation/word_segmentation_buckeye.py representations/$model/buckeye_$data_split_sample1/contextualized/frame_level/ data_samples/buckeye/segmentation/buckeye_$data_split.tsv $layer $prominence $dist $window_size $result_dir
```

The results (including precision, recall, F-score, and R-value) can then be evaluated with the scripts provided in Herman's repository.

### 4. Spoken sentence similarity
For extracting utterance level representations for spoken STS
```
for subset_id in 0 1; do
  bash scripts/extract_rep_spoken_sts.sh $model_name $ckpt_dir $subset_id $save_dir_pth $pckg_dir
done
```

Evaluate spoken STS, mean of all pairs of speaker combinations
```
bash scripts/eval_spoken_sts.sh $model_name $save_dir_pth $path_to_sts_data
```


## Acknowledgements
1. Thanks to Ju-Chieh Chou ([@jjery2243542](https://github.com/jjery2243542)) for help with testing the codebase.

2. Thanks to Lugosch et al. for making [Librispeech alignments](https://zenodo.org/record/2619474#.YnB_1fPMK3I) publicly available.
```
Loren Lugosch, Mirco Ravanelli, Patrick Ignoto, Vikrant Singh Tomar, and Yoshua Bengio, "Speech Model Pre-training for End-to-End Spoken Language Understanding", Interspeech 2019
Michael McAuliffe, Michaela Socolof, Sarah Mihuc, Michael Wagner, and Morgan Sonderegger, "Montreal Forced Aligner: trainable text-speech alignment using Kaldi", Interspeech 2017
```

3. Thanks to Shane Settle ([@shane-settle](https://github.com/shane-settle)) for providing acoustically grounded word embeddings trained on Librispeech data
```
Shane Settle, Kartik Audhkhasi, Karen Livescu, and Michael Picheny, “Acoustically grounded word embeddings for improved acoustics-to-word speech recognition”, in ICASSP, 2019
```

4. Thanks to Raghu et al. for making their [CCA implementation](https://github.com/google/svcca) publicly available. We use their library with some modifications and corrections. 
```
Maithra Raghu, Justin Gilmer, Jason Yosinski, and Jascha Sohl-Dickstein, "SVCCA: Singular Vector Canonical Correlation Analysis for Deep Learning Dynamics and Interpretability", NeurIPS 2017
Ari S. Morcos, Maithra Raghu, and Samy Bengio, "Insights on Representational Similarity in Deep Neural Networks with Canonical Correlation", NeurIPS 2018
```

5. Thanks to Yulia Tsvetkov ([@ytsvetko](https://github.com/ytsvetko)) for making the [semantic and syntactic attributes](https://github.com/ytsvetko/qvec/tree/master/oracles) publicly available. 
```
Yulia Tsvetkov, Manaal Faruqui, and Chris Dyer, "Correlation-based intrinsic evaluation of word vector representations", 1st Workshop on Evaluating Vector-Space Representations for NLP, 2016
Yulia Tsvetkov, Manaal Faruqui, Wang Ling, Guillaume Lample, and Chris Dyer, "Evaluation of word vector representations by subspace alignment", EMNLP, 2016.
```
