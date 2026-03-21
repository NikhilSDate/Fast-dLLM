export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true

export MASTER_ADDR=localhost
export MASTER_PORT=29500

task=gsm8k
length=256
block_length=32
num_fewshot=5
steps=$((length / block_length))

accelerate launch --debug llada/eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} \
--confirm_run_unsafe_code --model llada_dist \
--model_args model_path='GSAI-ML/LLaDA-8B-Instruct',gen_length=${length},steps=${length},block_length=${block_length},show_speed=True