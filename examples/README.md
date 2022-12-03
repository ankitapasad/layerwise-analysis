`recipe.sh` demonstrates an end-to-end usage of this codebase for experiments on cca-mel, mi-phone, and cca-phone.

The `for ...` loops in the script are written so only to give the user an idea of what constitutes a complete workflow. The script will run fine as is but will be very slow to execute as everything is processed serially. **The individual commands within the loops can be and are encouraged to be run parallely.** The steps themselves should be executed in the ordered fashion, i.e. step 2 before step 3, step 3 before step 4.

Before running the recipe:
1. Make sure that the paths are set correctly to reflect
- Librispeech data location (`path_to_librispeech_data`)
- Location where you want the alignment files to be saved(`alignment_data_dir`) -- it takes _6Gb_ of space
- Location where the extracted representation files should be saved (`save_dir_pth`) -- the representations extracted form each dev-clean sample set take upto _12Gb_ and for each train-clean sample set _23Gb_.

2. You can change the `steps` variable to skip the steps previously done. For example, if you already have the alignment files saved at `alignment_data_dir`, set `steps=2` so that the script skips that step and goes over to the data sampling and extraction step (step 2)
