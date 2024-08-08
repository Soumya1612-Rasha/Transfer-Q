# Transfer Q*: Principled Decoding for LLM Alignment

## Setup
The packages and versions used are mentioned in requirements.txt

conda create -n tq python=3.9 -y
conda activate tq

cd transfer_q
pip -r requirements.txt

# For direct transfer tasks on HH-RLHF dataset run the following command:

python collect_model_outs.py --run_percent 100. --config="example.config" --out_file="run_outs/example_out" --dataset="Dahoas/full-hh-rlhf" --task_type direct

# For indirect transfer tasks on HH-RLHF run the following command:

python collect_model_outs.py --run_percent 100. --config="example.config" --out_file="run_outs/example_out" --dataset="Dahoas/full-hh-rlhf" --task_type indirect

# To measure reward of generated responses run the following command:

python measure_reward.py --out_file="run_outs/example_out_0.jsonl"