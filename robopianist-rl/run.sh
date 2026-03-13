#!/bin/bash

PYTHON="YOUR_PYTHON_PATH"

WANDB_DIR= MUJOCO_GL=egl XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_VISIBLE_DEVICES=0 MUJOCO_EGL_DEVICE_ID=0 $PYTHON train.py \
    --mode online \
    --entity "YOUR_WANDB_ENTITY" \
    --project "YOUR_WANDB_PROJECT" \
    --warmstart-steps 5000 \
    --max-steps 5000000 \
    --discount 0.8 \
    --agent-config.critic-dropout-rate 0.01 \
    --agent-config.critic-layer-norm \
    --agent-config.hidden-dims 256 256 256 \
    --trim-silence \
    --gravity-compensation \
    --reduced-action-space \
    --control-timestep 0.05 \
    --n-steps-lookahead 10 \
    --environment-name 'YOUR_ENVIRONMENT_NAME' \
    --action-reward-observation \
    --primitive-fingertip-collisions \
    --disable-fingering-reward \
    --enable-tactile \
    --enable-velocity-reward \
    --velocity-reward-start-step 1000000 \
    --eval-episodes 1 \
    --camera-id "piano/back" \
    --tqdm-bar
