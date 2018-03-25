# cycle-gan
A pytorch implementation of [cycle gan](https://arxiv.org/pdf/1703.10593.pdf). 

## Introduction 
<iframe width="560" height="315" src="https://www.youtube.com/embed/D4C1dB9UheQ" frameborder="0" allow="autoplay; encrypted-media" allowfullscreen></iframe>

## Dependencies 
- python3
- pytorch(GPU/CPU)
- torchvision
- Pillow


## Google Colab Notebook
Google has provided colab notebook with free gpu instances to experiment with deep learning. For more details, [here](https://colab.research.google.com/notebooks/welcome.ipynb).

To setup and run complete package use `CycleGan.ipynb`. It has all the cells to download
- upload the notebook to Gdrive
- open it in Colab
- set the runtime to use gpu 
- uncomment the cells to install `pytorch`
- uncommnet the cells to download dataset. 

## MNIST-SVHN Cycle GAN

To run 
```
python main.py
```


## Acknowledgments
- "Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks", Zhu et al. 2017. 
```
@inproceedings{CycleGAN2017,
  title={Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networkss},
  author={Zhu, Jun-Yan and Park, Taesung and Isola, Phillip and Efros, Alexei A},
  booktitle={Computer Vision (ICCV), 2017 IEEE International Conference on},
  year={2017}
}
```
- [Official Pytorch Implementation](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)
- [Minimal Pytorch Implementation](https://github.com/yunjey/mnist-svhn-transfer)
- [Clean and Readable Implementation](https://github.com/aitorzip/PyTorch-CycleGAN)