## 1 Install

执行以下命令：

```shell
conda create -n cnnlstm python=3.7
conda activate cnnlstm
pip3 install torch==1.8.2 torchvision==0.9.2 torchaudio==0.8.2 --extra-index-url https://download.pytorch.org/whl/lts/1.8/cu111
pip install -r requirements.txt

```

## 2 Training

```shell
python main.py
```

### 注意
1.  batch_size的设定，在调试时用1，在分布式训练时有多少卡就设为多少，且需要修改以下代码

```python
trainLoader = torch.utils.data.DataLoader(
        trainvalFolder,
        # batch_size=1, # debug时用的
        batch_size=args.batch_size // torch.cuda.device_count(), # 分布式训练必须的
        shuffle=train_sampler is None,
        num_workers=nw,
        sampler=train_sampler,
        pin_memory=True,
        drop_last=True,
    )
```

## 3 Testing

- 带有分类分支的测试代码： 
```shell
python test.py
```

## 4 多GPU训练

```shell
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 python -m torch.distributed.launch --nproc_per_node=6 main.py --device 0,1,2,3,4,5
```
