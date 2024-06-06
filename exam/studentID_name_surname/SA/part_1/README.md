**Important notes**

New conda environment yaml file is provided with the additional dependency of wandb package.

```bash
conda env create -f nlu24_project_env.yaml -n nlu24
conda activate nlu24
```

The following code scripts and files were obtained from the repository [E2E-TBSA](https://github.com/lixin4ever/E2E-TBSA/tree/master):

- `utils_e2e_tbsa.py` is equivalent to `utils.py` in the original repository. 
- `glue_utils.py` 
- `mpqa_full.txt` 
- `seq_utils.py`
- **data** folder

The reason to have a copy of these files in this repository is so that we have access to the data and evaluation functions.