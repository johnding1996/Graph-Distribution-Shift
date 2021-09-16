#!/usr/bin/env bash

# Lines that begin with #SBATCH specify commands to be used by SLURM for scheduling
#SBATCH --job-name=graph-dg                           # sets the job name if not set from environment
#SBATCH --array=0                                   # Submit 8 array jobs, throttling to 4 at a time
#SBATCH --output slurm-logs/%x_%A_%a.log                # indicates a file to redirect STDOUT to; %j is the jobid, _%A_%a is array task id
#SBATCH --error slurm-logs/%x_%A_%a.log                 # indicates a file to redirect STDERR to; %j is the jobid,_%A_%a is array task id
#SBATCH --time=24:00:00                                 # how long you think your job will take to complete; format=hh:mm:ss
#SBATCH --account=scavenger                             # set QOS, this will determine what resources can be requested
#SBATCH --qos=scavenger                                 # set QOS, this will determine what resources can be requested
#SBATCH --partition=scavenger
#SBATCH --gres=gpu:rtx2080ti:1
#SBATCH --exclude=cmlgrad02,cml12,cmlgrad05
#SBATCH --cpus-per-task=4
#SBATCH --mem 16gb                                      # memory required by job; if unit is not specified MB will be assumed
#SBATCH --nice=0
#SBATCH --mail-type=None                                # Valid type values are NONE, BEGIN, END, FAIL, REQUEUE,


function runexp {

device=${1}
dataset=${2}
algorithm=${3}
model=${4}
root_dir=${5}

log_path=${root_dir}/sanity_check_logs/${dataset}-${algorithm}-${model}.log

python run_expt.py --device ${device} --dataset ${dataset}  --algorithm ${algorithm} --model ${model}  --root_dir ${root_dir} 2> ${log_path}

#> ${log_path} 2>&1
}

device=0
datasets=( ogb-molhiv    ) #4
algorithms=( deepCORAL )            #5
models=( gcn_virtual  ) #6
root_dir=/cmlscratch/kong/datasets/graph_domain

id=${SLURM_ARRAY_TASK_ID}

#runexp   device          dataset                   algorithm                       model            root_dir
runexp  ${device}   ${datasets[$id]}   ${algorithms[$id]}  ${models[$id]}  ${root_dir}

