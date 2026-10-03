import albumentations as A
from .config import IMAGE_WIDTH, IMAGE_HEIGHT


def train_transform():
    return A.Compose([
        A.Resize(IMAGE_HEIGHT, IMAGE_WIDTH),

        A.Affine(
            scale=(0.90, 1.10),
            translate_percent=(-0.06, 0.06),
            rotate=(-5, 5),
            shear=(-3, 3),
            border_mode=0,
            fill=0,
            p=0.75,
        ),

        A.Perspective(
            scale=(0.02, 0.07),
            keep_size=True,
            fit_output=False,
            fill=0,
            p=0.35,
        ),

        A.OneOf([
            A.RandomBrightnessContrast(
                brightness_limit=0.30,
                contrast_limit=0.30,
                p=1.0,
            ),
            A.RandomGamma(
                gamma_limit=(65, 145),
                p=1.0,
            ),
        ], p=0.75),

        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 7), p=1.0),
            A.MotionBlur(blur_limit=(3, 9), p=1.0),
            A.Defocus(radius=(2, 5), alias_blur=(0.1, 0.5), p=1.0),
        ], p=0.35),

        A.GaussNoise(
            std_range=(0.01, 0.09),
            mean_range=(0.0, 0.0),
            p=0.45,
        ),

        A.ImageCompression(
            quality_range=(45, 100),
            p=0.25,
        ),
    ])


def eval_transform():
    return A.Compose([
        A.Resize(IMAGE_HEIGHT, IMAGE_WIDTH),
    ])


def offline_augmentation_transform():
    return A.Compose([
        A.Affine(
            scale=(0.88, 1.12),
            translate_percent=(-0.07, 0.07),
            rotate=(-6, 6),
            shear=(-4, 4),
            border_mode=0,
            fill=0,
            p=0.9,
        ),

        A.Perspective(
            scale=(0.015, 0.08),
            keep_size=True,
            fit_output=False,
            fill=0,
            p=0.5,
        ),

        A.OneOf([
            A.RandomBrightnessContrast(
                brightness_limit=0.35,
                contrast_limit=0.35,
                p=1.0,
            ),
            A.RandomGamma(
                gamma_limit=(60, 155),
                p=1.0,
            ),
        ], p=0.85),

        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 7), p=1.0),
            A.MotionBlur(blur_limit=(3, 9), p=1.0),
            A.Defocus(radius=(2, 5), alias_blur=(0.1, 0.5), p=1.0),
        ], p=0.45),

        A.GaussNoise(
            std_range=(0.01, 0.10),
            mean_range=(0.0, 0.0),
            p=0.55,
        ),

        A.ImageCompression(
            quality_range=(40, 100),
            p=0.30,
        ),
    ])
