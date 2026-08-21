#!/bin/bash
set -e
logs_folder="logs"
mkdir -p $logs_folder
# Run the script

seed_max=2

# Change this to your name (used as prefix for experiment names).
user_name="jason"

## MAJOR ARGUMENTS TO CHECK!
# GPU number (check unused GPU with nvidia-smi)
cuda_device=1
# "double_integrator" or "airtaxi"
dynamics_type="airtaxi"
scenario_name="navigation_graph_safe"
n_agents=4
seed=0

# Your wandb account/team entity. Get it with: python -c "import wandb; print(wandb.Api().default_entity)"
wandb_entity="1205456072-southeast-university"

if [ "$dynamics_type" == "double_integrator" ]; then
    episode_length=250
    world_size=4
    n_landmarks=2
    num_env_steps=5000000
elif [ "$dynamics_type" == "airtaxi" ]; then
    episode_length=350
    world_size=6
    n_landmarks=2
    num_env_steps=5000000
else
    echo "Error: Unsupported dynamics type '$dynamics_type'"
    exit 1
fi
num_internal_step=1

if [ "$dynamics_type" == "kinematic_vehicle" ]; then
    str_dynamics_type="kv"
elif [ "$dynamics_type" == "double_integrator" ]; then
    str_dynamics_type="di"
elif [ "$dynamics_type" == "airtaxi" ]; then
    str_dynamics_type="airtaxi"
else
    echo "Error: Unsupported dynamics type '$dynamics_type'"
    exit 1
fi

str_scenario="random"
if [ "$scenario_name" == "navigation_graph_safe_eval" ]; then
    str_scenario="eval"
    num_env_steps=6000000
fi

# scripts/train_mpe.py saves runs to <parent-of-project>/results/<scenario>/<experiment>.
# Compute that directory from this script's location so phase 2 can find the warmstart model.
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_ROOT="$(dirname "${PROJECT_ROOT}")/results"
echo "PROJECT_ROOT: ${PROJECT_ROOT}"
echo "RESULTS_ROOT: ${RESULTS_ROOT}"

# Run a single training phase.
#   $1 = use_safety_filter  ("True"/"False")
#   $2 = str_safety_filter  ("1"/"0", used in wandb project name)
#   $3 = model_dir_arg      ("" or "--model_dir=<path to warmstart model>")
run_training() {
    local use_safety_filter=$1
    local str_safety_filter=$2
    local model_dir_arg=$3
    local datetime_str=$(date '+%y%m%d_%H%M%S')
    local experiment_name_str="${user_name}_${datetime_str}_agent${n_agents}_landmark${n_landmarks}_eplength${episode_length}_world${world_size}"
    local num_rollout_threads=32

    echo "============================================================"
    echo "Running training: use_safety_filter=${use_safety_filter} ${model_dir_arg}"
    echo "project: ${str_dynamics_type}_env_${str_scenario}_safety_${str_safety_filter}"
    echo "experiment: ${experiment_name_str}"
    echo "============================================================"
    CUDA_VISIBLE_DEVICES=${cuda_device} \
    python -u scripts/train_mpe.py --use_valuenorm --use_popart \
    --project_name "${str_dynamics_type}_env_${str_scenario}_safety_${str_safety_filter}" \
    --env_name "GraphMPE" \
    --algorithm_name "rmappo" \
    --seed ${seed} \
    --experiment_name ${experiment_name_str} \
    --scenario_name ${scenario_name} \
    --dynamics_type ${dynamics_type} \
    --num_agents=${n_agents} \
    --num_landmarks=${n_landmarks} \
    --n_training_threads 1 --n_rollout_threads ${num_rollout_threads} \
    --num_mini_batch 1 \
    --episode_length ${episode_length} \
    --num_env_steps ${num_env_steps} \
    --ppo_epoch 10 --use_ReLU --gain 0.01 --lr 7e-4 --critic_lr 7e-4 \
    --user_name "${wandb_entity}" \
    --use_cent_obs "False" \
    --graph_feat_type "relative" \
    --use_dones "False" \
    --collaborative "False" \
    --num_walls 0 \
    --world_size=${world_size} \
    --auto_mini_batch_size --target_mini_batch_size 4096 \
    --use_safety_filter ${use_safety_filter} \
    --num_internal_step ${num_internal_step} \
    --use_masking "True" \
    --checkpoint_interval 50 \
    ${model_dir_arg}
}

# ===================== PHASE 1: warmstart (no safety filter, from scratch) =====================
run_training "False" "0" ""

# Locate the warmstart model: most recent actor.pt saved under RESULTS_ROOT (wandb run dir).
warmstart_model_dir=$(find "${RESULTS_ROOT}" -name actor.pt -printf '%T@ %h\n' 2>/dev/null | sort -rn | head -1 | cut -d' ' -f2)
if [ -z "$warmstart_model_dir" ]; then
    echo "ERROR: warmstart model not found under ${RESULTS_ROOT}. Cannot start phase 2."
    exit 1
fi
echo "warmstart model dir: ${warmstart_model_dir}"

# ===================== PHASE 2: safety-informed (with safety filter) =====================
# Enable POTENTIAL_CONFLICT reward (our method) in multiagent/config.py before phase 2.
sed -i '/POTENTIAL_CONFLICT = False/s/False/True/' multiagent/config.py
echo "Set RewardBinaryConfig.POTENTIAL_CONFLICT = True"

run_training "True" "1" "--model_dir=${warmstart_model_dir}"

# Revert config.py to default.
sed -i '/POTENTIAL_CONFLICT = True/s/True/False/' multiagent/config.py
echo "Reverted RewardBinaryConfig.POTENTIAL_CONFLICT = False"
echo "All training phases complete."
