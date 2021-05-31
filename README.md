# Dataset
```wget https://www.statmt.org/wmt10/training-giga-fren.tar```

### Command
```python -m torch.distributed.launch --nproc_per_node=4 --nnodes=1 --master_addr=127.0.0.1 --master_port=9095 --node_rank 0 train.py -c configs/model/resnet50x1.yaml configs/dataset/cifar10.yaml configs/optimizer/adamw.yaml configs/scheduler/multistep.yaml configs/loss/ce.yaml --use_amp```
