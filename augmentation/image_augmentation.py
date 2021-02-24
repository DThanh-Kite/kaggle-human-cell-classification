import imgaug as ia
ia.seed(8198741)
from imgaug import augmenters as iaa
from loguru import logger


# TODO: replace imgaug with albumetation
def get_augmentation(version):
    augmentation = None
    if version is None:
        return augmentation
    logger.info('Using augmentation version {}'.format(version))
    if version == '1.0':
        sometimes = lambda aug: iaa.Sometimes(0.5, aug)
        augmentation = iaa.Sequential([
            sometimes(iaa.Affine(
                scale={"x": (0.8, 1.2), "y": (0.8, 1.2)}, # scale images to 80-120% of their size, individually per axis
                translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)}, # translate by -20 to +20 percent (per axis)
                rotate=(-45, 45), # rotate by -45 to +45 degrees
                shear=(-16, 16), # shear by -16 to +16 degrees
                order=[0, 1], # use nearest neighbour or bilinear interpolation (fast)
                mode=ia.ALL
            )),
            iaa.GammaContrast(gamma=(0.5, 1.75)),
        ])

    elif version == '1.1':
        sometimes = lambda aug: iaa.Sometimes(0.3, aug)
        augmentation = iaa.Sequential([
            sometimes(iaa.Affine(
                scale={"x": (0.8, 1.2), "y": (0.8, 1.2)}, # scale images to 80-120% of their size, individually per axis
                translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)}, # translate by -20 to +20 percent (per axis)
                rotate=(-45, 45), # rotate by -45 to +45 degrees
                shear=(-16, 16), # shear by -16 to +16 degrees
                order=[0, 1], # use nearest neighbour or bilinear interpolation (fast)
                mode=ia.ALL # use any of scikit-image's warping modes (see 2nd image from the top for examples)
            )),
            iaa.GammaContrast(gamma=(0.5, 1.75)),
            sometimes(iaa.CropAndPad(percent=(-0.05, 0.1),
                                    pad_mode=ia.ALL,
                                    pad_cval=(0, 255)
            ))
        ])

    elif version == '1.2':
        sometimes = lambda aug: iaa.Sometimes(0.5, aug)
        augmentation = iaa.Sequential([
            sometimes(iaa.Affine(
                scale={"x": (0.8, 1.2), "y": (0.8, 1.2)}, # scale images to 80-120% of their size, individually per axis
                translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)}, # translate by -20 to +20 percent (per axis)
                rotate=(-15, 15), # rotate by -45 to +45 degrees
                shear=(-16, 16), # shear by -16 to +16 degrees
                order=[0, 1], # use nearest neighbour or bilinear interpolation (fast),
                cval=(0, 255),
                mode='constant' # use any of scikit-image's warping modes (see 2nd image from the top for examples)
            )),
            iaa.OneOf([
                iaa.GammaContrast(gamma=(0.5, 1.75)),
                iaa.Multiply((0.5, 1.5)),
                iaa.LinearContrast((0.25, 1.5))
            ]),
            sometimes(iaa.CropAndPad(percent=(-0.05, 0.1),
                                     pad_mode=ia.ALL,
                                     pad_cval=(0, 255)
            )),
            iaa.Fliplr(p=0.5),
            iaa.Flipud(p=0.5),
        ])

    elif version == '1.3':
        sometimes = lambda aug: iaa.Sometimes(0.3, aug)
        augmentation = iaa.Sequential([
            iaa.Resize({"height": 224, "width": 224}),
            iaa.Affine(
                scale={"x": (0.8, 1.2), "y": (0.8, 1.2)}, # scale images to 80-120% of their size, individually per axis
                translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)}, # translate by -20 to +20 percent (per axis)
                rotate=(-45, 45), # rotate by -45 to +45 degrees
                order=[0, 1], # use nearest neighbour or bilinear interpolation (fast)
                mode=ia.ALL
            ),
            iaa.OneOf([
                iaa.Dropout((0.01, 0.1), per_channel=0.5), # randomly remove up to 10% of the pixels
                iaa.CoarseDropout((0.03, 0.15), size_percent=(0.02, 0.05), per_channel=0.2),
            ]),
            iaa.Fliplr(p=0.5),
            iaa.Flipud(p=0.5),
            iaa.LinearContrast((0.5, 2.0), per_channel=0.5),
            # sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.05))), # sometimes move parts of the image around
            # sometimes(iaa.PerspectiveTransform(scale=(0.01, 0.1)))
        ])
    elif version == '1.4':
        augmentation = iaa.Sequential([
            iaa.Affine(
                scale={"x": (0.8, 1.2), "y": (0.8, 1.2)}, # scale images to 80-120% of their size, individually per axis
                translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)}, # translate by -20 to +20 percent (per axis)
                rotate=(-45, 45), # rotate by -45 to +45 degrees
                mode=ia.ALL
            ),
            iaa.Fliplr(p=0.5),
            iaa.Flipud(p=0.5),
        ])
    elif version == '1.5':
        sometimes = lambda aug: iaa.Sometimes(0.5, aug)
        augmentation = iaa.Sequential([
            iaa.Affine(
                scale={"x": (0.8, 1.2), "y": (0.8, 1.2)}, # scale images to 80-120% of their size, individually per axis
                translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)}, # translate by -20 to +20 percent (per axis)
                rotate=(-45, 45), # rotate by -45 to +45 degrees
                mode=ia.ALL
            ),
            iaa.Fliplr(p=0.5),
            iaa.Flipud(p=0.5),
            iaa.OneOf([
                    iaa.GammaContrast(gamma=(0.5, 1.75)),
                    iaa.Multiply((0.5, 1.5)),
                    iaa.LinearContrast((0.25, 1.5))
                ]),
                sometimes(iaa.CropAndPad(percent=(-0.05, 0.1),
                                        pad_mode=ia.ALL,
                                        pad_cval=(0, 255)
                )),
        ])
    else:
        # Make sure that augmentation is used
        raise "Unsupported version of augmentation method ({})".format(version)

    return augmentation