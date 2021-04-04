#  [Designing a Practical Degradation Model for Deep Blind Image Super-Resolution](https://arxiv.org/pdf/2103.14006.pdf)
[Kai Zhang](https://cszn.github.io/), Jingyun Liang, [Luc Van Gool](https://vision.ee.ethz.ch/people-details.OTAyMzM=.TGlzdC8zMjQ4LC0xOTcxNDY1MTc4.html), [Radu Timofte](http://people.ee.ethz.ch/~timofter/)  
_[Computer Vision Lab](https://vision.ee.ethz.ch/the-institute.html), ETH Zurich, Switzerland_

[[Paper](https://arxiv.org/abs/2103.14006)]

✨ _**Some visual examples**_: [oldphoto2](https://imgsli.com/NDgzMjU); [butterfly](https://imgsli.com/NDgyNjY); [comic](https://imgsli.com/NDgyNzg); [oldphoto3](https://imgsli.com/NDgyNzk); [oldphoto6](https://imgsli.com/NDgyODA)
___________

* [Testing code](#testing-code)
* [Main idea](#main-idea)
* [Comparison](#comparison)
* [More visual results on RealSRSet dataset](#more-visual-results-on-realsrset-dataset)
* [Visual results on DPED dataset](#visual-results-on-dped-dataset)
* [Citation](#citation)
* [Acknowledgments](#acknowledgments)

Testing code
----------

* [main_test_bsrgan.py](main_test_bsrgan.py)
* [model_zoo](model_zoo) (_Download the following models from [Google drive](https://drive.google.com/drive/folders/13kfr3qny7S2xwG9h7v95F5mkWs0OmU0D?usp=sharing) or [腾讯微云](https://share.weiyun.com/5qO32s3)_).
  * RRDB.pth  --->  [original link](https://github.com/xinntao/ESRGAN)
  * ESRGAN.pth --->   [original link](https://github.com/xinntao/ESRGAN)
  * FSSR_DPED.pth --->   [original link](https://github.com/ManuelFritsche/real-world-sr)
  * FSSR_DPED.pth --->   [original link](https://github.com/ManuelFritsche/real-world-sr)
  * RealSR_DPED.pth --->   [original link](https://github.com/jixiaozhong/RealSR)
  * RealSR_JPEG.pth --->   [original link](https://github.com/jixiaozhong/RealSR)
  * BSRNet.pth 🌱
  * BSRGAN.pth 🌱

Main idea
----------

<img src="figs/degradationmodel.png" width="790px"/> 

__Design a new degradation model to synthesize LR images for training:__

* **_1) Make the blur, downsampling and noise more practical_**
  * **_Blur:_** _two convolutions with isotropic and anisotropic Gaussian kernels from both the HR space and LR space_
  * **_Downsampling:_** _nearest, bilinear, bicubic, down-up-sampling_
  * **_Noise:_** _Gaussian noise, JPEG compression noise, processed camera sensor noise_
* **_2) Degradation shuffle:_** _instead of using the commonly-used blur/downsampling/noise-addition pipeline, we perform randomly shuffled degradations to synthesize LR images_

__Some notes on the proposed degradation model:__

* *The degradation model is mainly designed to synthesize degraded LR images. Its most direct application is to train a deep blind super-resolver with paired LR/HR images. In particular, the degradation model can be performed on a large dataset of HR images to produce unlimited perfectly aligned training images, which typically do not suffer from the limited data issue of laboriously collected paired data and the misalignment issue of unpaired training data.*
 
* *The degradation model tends to be unsuited to model a degraded LR image as it involves too many degradation parameters and also adopts a random shuffle strategy.*

* *The degradation model can produce some degradation cases that rarely happen in real-world scenarios, while this can still be expected to improve the generalization ability of the trained deep blind super-resolver.*

* *A DNN with large capacity has the ability to handle different degradations via a single model. This has been validated multiple times. For example, DnCNN is able
to handle SISR with different scale factors, JPEG compression deblocking with different quality factors and denoising for a wide range of noise levels, while still having a performance comparable to VDSR for SISR. It is worth noting that even when the super-resolver reduces the performance for unrealistic bicubic downsampling, it is still a preferred choice for real SISR.*

* *One can conveniently modify the degradation model by changing the degradation parameter settings and adding more reasonable degradation
types to improve the practicability for a certain application.*




Comparison
----------


<img src="figs/comparison.png" width="790px"/> 

*These no-reference IQA metrics, i.e., NIQE, NRQM and PI, do not always match perceptual visual quality [1] and the IQA metric should be updated with new SISR methods [2]. We further argue that the IQA metric for SISR should also be updated with new image degradation types, which we leave for future work.*

```
[1] "NTIRE 2020 challenge on real-world image super-resolution: Methods and results." CVPRW, 2020.
[2] "PIPAL: a large-scale image quality assessment dataset for perceptual image restoration." ECCV, 2020.
```



More visual results on [RealSRSet](testsets/RealSRSet) dataset
----------


**Left**: [real images](https://github.com/cszn/BSRNet/tree/main/testsets/RealSRSet) **|** **Right**: [super-resolved images with scale factor 4](https://github.com/cszn/BSRNet/tree/main/testsets/BSRGAN)

<img src="testsets/RealSRSet/butterfly.png" width="156px"/> <img src="testsets/BSRGAN/butterfly_BSRGAN.png" width="624px"/>

<img src="testsets/RealSRSet/oldphoto2.png" width="156px"/> <img src="testsets/BSRGAN/oldphoto2_BSRGAN.png" width="624px"/>

<img src="testsets/RealSRSet/oldphoto2.png" width="390px"/> <img src="testsets/BSRGAN/oldphoto2_BSRGAN.png" width="390px"/>

<img src="testsets/RealSRSet/oldphoto3.png" width="156px"/> <img src="testsets/BSRGAN/oldphoto3_BSRGAN.png" width="624px"/>

<img src="testsets/RealSRSet/oldphoto3.png" width="390px"/> <img src="testsets/BSRGAN/oldphoto3_BSRGAN.png" width="390px"/>

<img src="testsets/RealSRSet/oldphoto6.png" width="390px"/> <img src="testsets/BSRGAN/oldphoto6_BSRGAN.png" width="390px"/>

<img src="testsets/RealSRSet/dog.png" width="390px"/> <img src="testsets/BSRGAN/dog_BSRGAN.png" width="390px"/>

<img src="testsets/RealSRSet/dped_crop00061.png" width="390px"/> <img src="testsets/BSRGAN/dped_crop00061_BSRGAN.png" width="390px"/>

<img src="testsets/RealSRSet/chip.png" width="390px"/> <img src="testsets/BSRGAN/chip_BSRGAN.png" width="390px"/>

<img src="testsets/RealSRSet/comic1.png" width="390px"/> <img src="testsets/BSRGAN/comic1_BSRGAN.png" width="390px"/>

<img src="testsets/RealSRSet/comic3.png" width="390px"/> <img src="testsets/BSRGAN/comic3_BSRGAN.png" width="390px"/>



Visual results on DPED dataset
----------

<img src="figs/00003.png" width="200px"/> <img src="figs/00003_BSRGAN.png" width="790px"/>

<img src="figs/00080.png" width="200px"/> <img src="figs/00080_BSRGAN.png" width="790px"/>

<img src="figs/00081.png" width="200px"/> <img src="figs/00081_BSRGAN.png" width="790px"/>

*Without using any prior information of DPED dataset for training, our BSRGAN still performs well.*




Citation
----------
```BibTex
@inproceedings{zhang2021designing,
  title={Designing a Practical Degradation Model for Deep Blind Image Super-Resolution},
  author={Zhang, Kai and Liang, Jingyun and Van Gool, Luc and Timofte, Radu},
  booktitle={arxiv},
  year={2021}
}
```


Acknowledgments
----------
This work was partly supported by the ETH Zurich Fund (OK), a Huawei Technologies Oy (Finland) project, and an Amazon AWS grant.


License
----------
This work is licensed under a Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License.

<!--
**_If you find this work is useful for real applications, I would be happy if you send an email to let me know (Kai Zhang: cskaizhang@gmail.com)._**
-->
