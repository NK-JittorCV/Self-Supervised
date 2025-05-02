# Sel-supervised learning based on Jittor

## Getting Started

### 1. Install Jittor
```bash
pip install jittor
python -m jittor.test.test_example
# If your computer contains an Nvidia graphics card, check the cudnn acceleration library
python -m jittor.test.test_cudnn_op
```
For more information on how to install jittor, you can check [here](https://cg.cs.tsinghua.edu.cn/jittor/download/).

### 2. Install OpenMPI
```bash
sudo apt install openmpi-bin openmpi-common libopenmpi-dev
```

For more information on how to use MPI for Distributed Training, please refer to [here](https://cg.cs.tsinghua.edu.cn/jittor/assets/docs/jittor.mpi.html).

## Supported methods

| Method | Documentation | Pytorch Link |
| ----------- | ----------- |----------- |
| SERE, TPAMI, 2023 | [documentation](docs/SERE.md) | https://github.com/MCG-NKU/SERE |
| HSSL, TPAMI, 2025 | [documentation](docs/HSSL.md) | https://github.com/lzyhha/HSSL |


Please refer to the above documentations of the specific methods for the training guidance.


## Citation
If this work is helpful for your research, please consider citing the following entry:
```
@article{li2023sere,
  title={SERE: Exploring Feature Self-relation for Self-supervised Transformer},
  author={Zhong-Yu Li and Shanghua Gao and Ming-Ming Cheng},
  journal=TPAMI,
  year={2023}
}

@article{li2025hssl,
  title={Enhancing Representations through Heterogeneous Self-Supervised Learning}, 
  author={Li, Zhong-Yu and Yin, Bo-Wen and Liu, Yongxiang and Liu, Li and Cheng, Ming-Ming},
  journal=TPAMI,
  year={2025}
}
```

## Acknowledgement

This repository is built using the [DINO](https://github.com/facebookresearch/dino) repository, the [iBOT](https://github.com/bytedance/ibot) repository, 
and the [MAE](https://github.com/facebookresearch/mae) repository.
