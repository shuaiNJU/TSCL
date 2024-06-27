## TSCL
When an Extra Rejection Class Meets Out-of-Distribution Detection in Long-tailed Image Classification


## Training

CIFAR10-LT: 

```
python train.py --gpu 0 --ds cifar10 --Lambda 1.0 --Lambda2 0.1 --T 0.1 --k 0.6 \
    --drp <where_you_store_all_your_datasets> --srp <where_to_save_the_ckpt>
```

CIFAR100-LT:

```
python train.py --gpu 0 --ds cifar100 --Lambda 1.0 --Lambda2 0.1 --T 0.1 --k 0.6 \
    --drp <where_you_store_all_your_datasets> --srp <where_to_save_the_ckpt>
```

