## TSCL
When an Extra Rejection Class Meets Out-of-Distribution Detection in Long-tailed Image Classification

## Overview Architecture
![image](https://github.com/shuaiNJU/TSCL/assets/48854550/d5992f37-fdd1-48bf-96c6-5492e5f06640)

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

## Testing

CIFAR10-LT:

```
for dout in texture svhn cifar tin lsun places365
do
python test.py --gpu 0 --ds cifar10 --dout $dout \
    --drp <where_you_store_all_your_datasets> \
    --ckpt_path <where_you_save_the_ckpt>
done
```

CIFAR100-LT:

```
for dout in texture svhn cifar tin lsun places365
do
python test.py --gpu 0 --ds cifar100 --dout $dout \
    --drp <where_you_store_all_your_datasets> \
    --ckpt_path <where_you_save_the_ckpt>
done
```
## Acknowledgement

Part of our codes are adapted from these repos:

PASCL- https://github.com/amazon-science/long-tailed-ood-detection

## Citation
@article{feng2024extra,
  title={When an extra rejection class meets out-of-distribution detection in long-tailed image classification},
  author={Feng, Shuai and Wang, Chongjun},
  journal={Neural Networks},
  pages={106485},
  year={2024},
  publisher={Elsevier}
}
