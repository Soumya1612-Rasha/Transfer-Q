# Transfer Q*: Principled Decoding for LLM Alignment

This codebase provides a Pytorch implementation for the paper: Transfer Q Star: Principled Decoding for LLM Alignment. 

## Setup
The packages and versions used are mentioned in requirements.txt
```
conda create -n tq python=3.9 -y
conda activate tq

cd transfer_q
mkdir run_outs
pip -r requirements.txt
```

# For direct transfer tasks on HH-RLHF dataset run the following command:

```
python collect_model_outs.py --run_percent 100. --config="example.config" --out_file="run_outs/example_out" --dataset="Dahoas/full-hh-rlhf" --task_type direct
```

# For indirect transfer tasks on HH-RLHF run the following command:

```
python collect_model_outs.py --run_percent 100. --config="example.config" --out_file="run_outs/example_out" --dataset="Dahoas/full-hh-rlhf" --task_type indirect
```
# To measure reward of generated responses run the following command:

```
python measure_reward.py --out_file="run_outs/example_out_0.jsonl"
```

## References

The codebase has been adapted from [ARGS](https://github.com/deeplearning-wisc/args).

## For bibtex citation 

```
@misc{chakraborty2024transferqstarprincipled,
      title={Transfer Q Star: Principled Decoding for LLM Alignment}, 
      author={Souradip Chakraborty and Soumya Suvra Ghosal and Ming Yin and Dinesh Manocha and Mengdi Wang and Amrit Singh Bedi and Furong Huang},
      year={2024},
      eprint={2405.20495},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2405.20495}, 
}
```
