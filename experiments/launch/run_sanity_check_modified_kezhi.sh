#!/usr/bin/env bash

# Lines that begin with #SBATCH specify commands to be used by SLURM for scheduling
<<<<<<< HEAD:experiments/launch/run_sanity_check_modified_kezhi.sh
#SBATCH --job-name=graph-dg                           # sets the job name if not set from environment
=======
#SBATCH --job-name=gds_test                             # sets the job name if not set from environment
>>>>>>> de6c851eee21f22ee39bd9d785ee3aad1a026c03:experiments/launch/run_sanity_check_modified.sh
#SBATCH --array=0-119                                   # Submit 8 array jobs, throttling to 4 at a time
#SBATCH --output slurm-logs/%x_%A_%a.log                # indicates a file to redirect STDOUT to; %j is the jobid, _%A_%a is array task id
#SBATCH --error slurm-logs/%x_%A_%a.log                 # indicates a file to redirect STDERR to; %j is the jobid,_%A_%a is array task id
#SBATCH --time=0:10:00                                  # how long you think your job will take to complete; format=hh:mm:ss
#SBATCH --account=scavenger                             # set QOS, this will determine what resources can be requested
#SBATCH --qos=scavenger                                 # set QOS, this will determine what resources can be requested
#SBATCH --partition=scavenger
#SBATCH --gres=gpu:rtx2080ti:1
#SBATCH --cpus-per-task=4
#SBATCH --mem 16gb                                      # memory required by job; if unit is not specified MB will be assumed
#SBATCH --nice=0
#SBATCH --mail-type=None                                # Valid type values are NONE, BEGIN, END, FAIL, REQUEUE,

function echoerr() { echo "$@" 1>&2; }

function runexp {

device=${1}
dataset=${2}
algorithm=${3}
model=${4}
root_dir=${5}

log_path=${root_dir}/sanity_check_logs/error.txt

(echoerr "Dataset: ${dataset}, Algorithm: ${algorithm}, Model: ${model}" && python run_expt.py --n_epochs 1 --device ${device} --dataset ${dataset}  --algorithm ${algorithm} --model ${model}  --root_dir ${root_dir}) 2>> ${log_path}

#> ${log_path} 2>&1
}

# ppa too large, skip for now
device=0
datasets=( ogb-molpcba ogb-molhiv ogbg-ppa RotatedMNIST ) #4
algorithms=( ERM deepCORAL groupDRO IRM FLAG )            #5
models=( gin gin_virtual gcn gcn_virtual cheb cheb_virtual ) #6
root_dir=/cmlscratch/kong/datasets/graph_domain

dataset_idx=$(( ${SLURM_ARRAY_TASK_ID} % 4 ))
algorithm_idx=$(( ${SLURM_ARRAY_TASK_ID} / 4 % 5 ))
model_idx=$(( ${SLURM_ARRAY_TASK_ID} / 20 % 6 ))


#runexp   device          dataset                   algorithm                       model            root_dir
runexp  ${device}   ${datasets[$dataset_idx]}   ${algorithms[$algorithm_idx]}  ${models[$model_idx]}  ${root_dir}