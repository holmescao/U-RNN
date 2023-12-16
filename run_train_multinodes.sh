#!/usr/bin/env bash

#--- Multi-nodes training hyperparams ---
master_addr="172.17.0.2"
any_unique_id=915795412

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 torchrun \
  --nproc_per_node=6 \
  --nnodes=1 \
  --node_rank=0 \
  --rdzv_id=$any_unique_id \
  --rdzv_backend=c10d \
  --rdzv_endpoint=${master_addr}:12345 \
  main.py -batch_size 6 --device 0,1,2,3,4,5

