<img src="https://github.com/Tencent/MedicalNet/blob/master/images/logo.png?raw=true" align=mid />

# MedicalNet Models
Unofficial repository to easily access MedicalNet models ([Med3D: Transfer Learning for 3D Medical Image Analysis](https://arxiv.org/abs/1904.00625)). 
In this repository we create entrypoints to make it easy to download MedicalNet models using torch [Hub](https://pytorch.org/docs/stable/hub.html).

Note: Resnet18 and Resnet34 not available since state_dict does not match with the ones from the original 
implementation.

## Downloading a pretrained model
```
import torch
model = torch.hub.load("Warvito/MedicalNet-models", 'medicalnet_resnet10')
```

## Citing MedicalNet
If you use this code or pre-trained models, please cite the following:
```
    @article{chen2019med3d,
        title={Med3D: Transfer Learning for 3D Medical Image Analysis},
        author={Chen, Sihong and Ma, Kai and Zheng, Yefeng},
        journal={arXiv preprint arXiv:1904.00625},
        year={2019}
    }
```

