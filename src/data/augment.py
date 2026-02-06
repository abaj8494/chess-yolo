"""Augmentation pipeline for chess piece detection training."""

import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_train_transforms(img_size: int = 640) -> A.Compose:
    """Get training augmentation pipeline.

    Heavy augmentations optimized for:
    - Angled camera views (perspective transforms)
    - Variable lighting conditions
    - Hand occlusion simulation
    - Different chess set appearances
    """
    return A.Compose(
        [
            # Resize
            A.LongestMaxSize(max_size=img_size),
            A.PadIfNeeded(
                min_height=img_size,
                min_width=img_size,
                border_mode=0,
                value=(114, 114, 114),
            ),
            # Geometric - critical for angled views
            A.Perspective(scale=(0.05, 0.15), p=0.5),
            A.Affine(
                rotate=(-15, 15),
                shear=(-10, 10),
                scale=(0.8, 1.2),
                p=0.5,
            ),
            A.HorizontalFlip(p=0.5),
            # Photometric - lighting variations
            A.OneOf(
                [
                    A.RandomBrightnessContrast(
                        brightness_limit=0.3, contrast_limit=0.3, p=1.0
                    ),
                    A.CLAHE(clip_limit=4.0, p=1.0),
                    A.ColorJitter(
                        brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=1.0
                    ),
                ],
                p=0.8,
            ),
            A.HueSaturationValue(
                hue_shift_limit=10, sat_shift_limit=30, val_shift_limit=20, p=0.4
            ),
            # Noise and blur - camera quality variations
            A.OneOf(
                [
                    A.GaussNoise(var_limit=(10, 50), p=1.0),
                    A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=1.0),
                    A.MultiplicativeNoise(multiplier=(0.9, 1.1), p=1.0),
                ],
                p=0.3,
            ),
            A.OneOf(
                [
                    A.MotionBlur(blur_limit=5, p=1.0),
                    A.GaussianBlur(blur_limit=5, p=1.0),
                    A.MedianBlur(blur_limit=5, p=1.0),
                ],
                p=0.2,
            ),
            # Occlusion simulation - player hands, objects
            A.CoarseDropout(
                max_holes=3,
                max_height=img_size // 8,
                max_width=img_size // 8,
                min_holes=1,
                min_height=img_size // 16,
                min_width=img_size // 16,
                fill_value=0,
                p=0.2,
            ),
            # Shadows and lighting effects
            A.RandomShadow(
                shadow_roi=(0, 0, 1, 1),
                num_shadows_limit=(1, 2),
                shadow_dimension=5,
                p=0.3,
            ),
            A.RandomToneCurve(scale=0.1, p=0.2),
            # Final normalization
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ],
        bbox_params=A.BboxParams(
            format="yolo",
            label_fields=["class_labels"],
            min_visibility=0.3,
        ),
    )


def get_val_transforms(img_size: int = 640) -> A.Compose:
    """Get validation/inference transforms (minimal augmentation)."""
    return A.Compose(
        [
            A.LongestMaxSize(max_size=img_size),
            A.PadIfNeeded(
                min_height=img_size,
                min_width=img_size,
                border_mode=0,
                value=(114, 114, 114),
            ),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ],
        bbox_params=A.BboxParams(
            format="yolo",
            label_fields=["class_labels"],
        ),
    )


def get_inference_transforms(img_size: int = 640) -> A.Compose:
    """Get inference-only transforms (no bbox params)."""
    return A.Compose(
        [
            A.LongestMaxSize(max_size=img_size),
            A.PadIfNeeded(
                min_height=img_size,
                min_width=img_size,
                border_mode=0,
                value=(114, 114, 114),
            ),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
        ]
    )
