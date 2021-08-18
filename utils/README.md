
# How to use the degradation model:
```python
from utils import utils_blindsr as blindsr
img_lq, img_hq = blindsr.degradation_bsrgan_plus(img, sf=4, use_shuffle=True, use_sharp=True, lq_patchsize=64)

```

# 
``` python
def degradation_bsrgan_plus(img, sf=4, use_shuffle=True, use_sharp=True, lq_patchsize=64):
    """
    This is an extended degradation model by combining
    the degradation models of BSRGAN and Real-ESRGAN
    ----------
    img: HXWXC, [0, 1], its size should be large than (lq_patchsizexsf)x(lq_patchsizexsf)
    sf: scale factor
    use_shuffle: the degradation shuffle
    use_sharp: sharpening the img

    Returns
    -------
    img: low-quality patch, size: lq_patchsizeXlq_patchsizeXC, range: [0, 1]
    hq: corresponding high-quality patch, size: (lq_patchsizexsf)X(lq_patchsizexsf)XC, range: [0, 1]
    """

    h1, w1 = img.shape[:2]
    img = img.copy()[:w1 - w1 % sf, :h1 - h1 % sf, ...]  # mod crop
    h, w = img.shape[:2]

    if h < lq_patchsize*sf or w < lq_patchsize*sf:
        raise ValueError(f'img size ({h1}X{w1}) is too small!')

    if use_sharp:
        img = add_sharpening(img)
    hq = img.copy()

    shuffle_order = random.sample(range(11), 11) if use_shuffle else range(11)

    for i in shuffle_order:
        if i == 0:
            img = add_blur(img, sf=sf)
        elif i == 1:
            img = add_resize(img, sf=4)
        elif i == 2:
            img = add_Gaussian_noise(img, noise_level1=2, noise_level2=25)
        elif i == 3:
            if random.random() > 0.9:
                img = add_Poisson_noise(img)
        elif i == 4:
            if random.random() > 0.9:
                img = add_speckle_noise(img)
        elif i == 5:
            img = add_JPEG_noise(img)
        elif i == 6:
            img = add_blur(img, sf=sf)
        elif i == 7:
            img = add_resize(img, sf=sf)
        elif i == 8:
            img = add_Gaussian_noise(img, noise_level1=2, noise_level2=25)
        elif i == 9:
            if random.random() > 0.9:
                img = add_Poisson_noise(img)
        elif i == 10:
            if random.random() > 0.9:
                img = add_speckle_noise(img)
        else:
            print('check the shuffle!')

    # resize to desired size
    img = cv2.resize(img, (int(1/sf*hq.shape[1]), int(1/sf*hq.shape[0])), interpolation=random.choice([1, 2, 3]))

    # add final JPEG compression noise
    img = add_JPEG_noise(img)

    # random crop
    img, hq = random_crop(img, hq, sf, lq_patchsize)

    return img, hq
```
