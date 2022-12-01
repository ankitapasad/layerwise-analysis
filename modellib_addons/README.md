This directory has model files with edits to return all intermediate layer representations. Each file is placed at the same relative path as in the original model package. 

You should replace the original files in the model packages with these edited versions before performing feature extraction.

Tip: If you are dealing with version change in your model package (when conpared to the last tested listed below), I still expect it to be easy and quick to port these changes. You'd have to make sure that the model file in question is still the same between versions and even if not the edits are minor and can be easily understood and manually added to the model files. 

## [fairseq](https://github.com/pytorch/fairseq): HuBERT and wav2vec2.0

- Last tested on [commit 0b54d9f](https://github.com/pytorch/fairseq/tree/0b54d9fb2e42c2f40db3449ca34586952b8abe94).
- If your fairseq is cloned at `fairseq_pckg_dir` and layerwise_analysis at `analysis_repo_dir`
```
commit_sha="0b54d9f"
cd $fairseq_pckg_dir
git checkout $commit_sha

# for wav2vec2.0
sub_dir=fairseq/models/wav2vec
cp $analysis_repo_dir/modellib_addons/fairseq_addons/$sub_dir/wav2vec2.py $fairseq_pckg_dir/$sub_dir/

# for HuBERT
sub_dir=fairseq/models/hubert
cp $analysis_repo_dir/modellib_addons/fairseq_addons/$sub_dir/hubert.py $fairseq_pckg_dir/$sub_dir/
```

## [av_hubert](https://github.com/facebookresearch/av_hubert): AV-HuBERT
- Last tested on [commit cd1fd24](https://github.com/facebookresearch/av_hubert/tree/cd1fd24e71b18f5c1a7203aec6ce4479a61e7e67).
- If your av_hubert is cloned at `av_hubert_pckg_dir` and layerwise_analysis at `analysis_repo_dir`
```
commit_sha="cd1fd24"
cd $av_hubert_pckg_dir
git checkout $commit_sha

sub_dir=fairseq/fairseq/models/wav2vec
cp $analysis_repo_dir/modellib_addons/av_hubert_addons/$sub_dir/wav2vec2.py $av_hubert_pckg_dir/$sub_dir/

sub_dir=avhubert
cp $analysis_repo_dir/modellib_addons/av_hubert_addons/$sub_dir/hubert.py $av_hubert_pckg_dir/$sub_dir/
```

## [FAST-VGS-Family](https://github.com/jasonppy/FaST-VGS-Family): FaST-VGS and FaST-VGS+
- Last tested on [commit c10b928](https://github.com/jasonppy/FaST-VGS-Family/tree/c10b928ee73c79c48290b28aa1e7f1bb5c1eb367).
- If your fast-vgs is cloned at `fast-vgs_pckg_dir`
```
commit_sha="c10b928"
cd $fast-vgs_pckg_dir
git checkout $commit_sha
```

## [wavlm](https://github.com/microsoft/unilm/tree/master/wavlm): WavLM 

- Last tested on [commit 67afeed](https://github.com/microsoft/unilm/tree/65f15af2a307ebb64cfb25adf54375b002e6fe8d/wavlm).
- If your unilm is cloned at `unilm_pckg_dir` and layerwise_analysis at `analysis_repo_dir`
```
commit_sha="67afeed"
cd $unilm_pckg_dir
git checkout $commit_sha

sub_dir=wavlm
cp $analysis_repo_dir/modellib_addons/unilm_addons/$sub_dir/WavLM.py $unilm_pckg_dir/$sub_dir/
```

## [fairseq](https://github.com/pytorch/fairseq): XLSR-53 and XLS-R

- Last tested on [commit ecea95c](https://github.com/facebookresearch/fairseq/tree/ce6c9eeae163ac04b79539c78e74f292f29eaa18).
- If your fairseq is cloned at `fairseq_pckg_dir` and layerwise_analysis at `analysis_repo_dir`
```
commit_sha="ecea95c"
cd $fairseq_pckg_dir
git checkout $commit_sha

sub_dir=fairseq/models/wav2vec
cp $analysis_repo_dir/modellib_addons/fairseq_xlsr_addons/$sub_dir/wav2vec2.py $fairseq_pckg_dir/$sub_dir/
```
