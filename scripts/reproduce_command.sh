# 1) Start a named tmux session on login node
tmux new -s fastdllm

# 2) Go to repo
cd /projects/bdes/ndate/Fast-dLLM

# 3) Allocate GPU shell (same as your srun script)
# bash scripts/srun.sh

# 4) Now you are on compute node: activate env + run eval
source /u/xzhou12/fastdllm_venv/bin/activate

# 5) Example: run Dream baseline with gen_length=512
GEN_LENGTH=512 bash scripts/auto_resume.sh dream baseline
GEN_LENGTH=256 bash scripts/auto_resume.sh dream prefix-cache-parallel
# Example: run LLaDA parallel with gen_length=512
GEN_LENGTH=512 bash scripts/auto_resume.sh llada parallel
GEN_LENGTH=256 bash scripts/auto_resume.sh llada prefix-cache-variable

ssh <delta-login-node>
tmux attach -t fastdllm
tmux a
tmux ls
tail -f /projects/bdes/ndate/Fast-dLLM/evals_results/<your-log-file>.log

tmux kill-session -t fastdllm
tmux kill-server # kills all sessions, use with caution!