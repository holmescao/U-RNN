CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 python -m torch.distributed.launch \
    --nproc_per_node=7 main.py \
    --device 1,2,3,4,5,6,7
