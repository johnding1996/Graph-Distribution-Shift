#!/usr/bin/env bash

# Lines that begin with #SBATCH specify commands to be used by SLURM for scheduling
#SBATCH --job-name=gds_batch                             # sets the job name if not set from environment
#SBATCH --array=0                                  # Submit 8 array jobs, throttling to 4 at a time
#SBATCH --output /cmlscratch/jkirchen/gds-root/gds-slurm-logs/%x_%A_%a.log                # indicates a file to redirect STDOUT to; %j is the jobid, _%A_%a is array task id
#SBATCH --error /cmlscratch/jkirchen/gds-root/gds-slurm-logs/%x_%A_%a.log                 # indicates a file to redirect STDERR to; %j is the jobid,_%A_%a is array task id
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

log_path=${root_dir}/batch-logs/error.txt

(echoerr "Dataset: ${dataset}, Algorithm: ${algorithm}, Model: ${model}" && \
python run_expt.py --seed 0 \
                   --n_epochs 1 \
                   --device ${device} \
                   --dataset ${dataset} \
                   --algorithm ${algorithm} \
                   --model ${model} \
                   --root_dir ${root_dir}/gds-data \
                   --log_dir ${root_dir}/gds-result-logs ) 2>> ${log_path}

#> ${log_path} 2>&1
}

# ppa too large, skip for now
device=0
# datasets=( ogb-molpcba ogb-molhiv ) #2
# algorithms=( ERM GCL )  #2
# models=( gin gcn ) #2

datasets=( ogb-molhiv ) #1
algorithms=( GCL )  #1
models=( gin ) #1

root_dir=/cmlscratch/jkirchen/gds-root #/gds-data

# dataset_idx=$(( ${SLURM_ARRAY_TASK_ID} % 2 ))
# algorithm_idx=$(( ${SLURM_ARRAY_TASK_ID} / 2 % 2 ))
# model_idx=$(( ${SLURM_ARRAY_TASK_ID} / 4 % 2 ))

dataset_idx=$(( ${SLURM_ARRAY_TASK_ID} % 1 ))
algorithm_idx=$(( ${SLURM_ARRAY_TASK_ID} / 2 % 1 ))
model_idx=$(( ${SLURM_ARRAY_TASK_ID} / 2 % 1 ))


#runexp   device          dataset                   algorithm                       model            root_dir
runexp  ${device}   ${datasets[$dataset_idx]}   ${algorithms[$algorithm_idx]}  ${models[$model_idx]}  ${root_dir}

# TODO test batch training of what we have, so we can run some stuff
