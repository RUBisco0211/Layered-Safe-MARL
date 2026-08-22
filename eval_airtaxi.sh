#!/bin/bash
set -e

export SAFE_MARL_HEADLESS_RENDER="${SAFE_MARL_HEADLESS_RENDER:-true}"

model_dir="trained_models/airtaxi_safety_informed"

python scripts/eval_mpe.py \
--model_dir="${model_dir}" \
--dynamics_type="airtaxi" \
--render_episodes=1 \
--num_agents=4 \
--num_obstacles=0 \
--num_landmarks=2 \
--seed=0 \
--episode_length=350 \
--use_dones=False \
--collaborative=False \
--scenario_name="navigation_graph_safe" \
--horizon=1 \
--save_gifs \
--use_render \
--num_walls=0 \
--world_size=6 \
--discrete_action=True \
--use_masking=True \
--use_safety_filter=True
