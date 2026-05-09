import torchvision.transforms as T

from config import Config


def build_ssl_transform(cfg: Config) -> T.Compose:
    """
    Aggressive two-view augmentation for BYOL pretraining on leaf images.

    Tuned for natural plant photography: strong color jitter to simulate
    lighting variation, Gaussian blur, random grayscale, and large random crops.
    Avoids stain-normalization (histopathology-specific) used in the cancer_detection
    sibling project.
    """
    s = cfg.ssl_color_jitter_strength
    # Leaf disease signals are color- and texture-sensitive:
    # hue/saturation reduced more than brightness/contrast per domain analysis
    return T.Compose([
        T.RandomResizedCrop(
            cfg.img_size,
            scale=(cfg.ssl_crop_scale_min, 1.0),
            interpolation=T.InterpolationMode.BICUBIC,
        ),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.5),
        T.RandomApply(
            [T.ColorJitter(
                brightness=0.7 * s,
                contrast=0.7 * s,
                saturation=0.5 * s,
                hue=0.1 * s,
            )],
            p=0.8,
        ),
        T.RandomGrayscale(p=0.1),
        T.RandomApply([T.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0))], p=0.2),
        T.ToTensor(),
        T.Normalize(mean=cfg.imagenet_mean, std=cfg.imagenet_std),
    ])


def build_train_transform(cfg: Config) -> T.Compose:
    """
    Moderate augmentation for supervised training on leaf images.

    Flips and mild color jitter without the aggressive transforms used for SSL,
    since disease labels are image-level and over-augmentation can obscure lesions.
    """
    return T.Compose([
        T.RandomResizedCrop(cfg.img_size, scale=(0.7, 1.0)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.5),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        T.ToTensor(),
        T.Normalize(mean=cfg.imagenet_mean, std=cfg.imagenet_std),
    ])


def build_val_transform(cfg: Config) -> T.Compose:
    """Deterministic resize + center crop for validation and inference."""
    return T.Compose([
        T.Resize(int(cfg.img_size * 1.14)),
        T.CenterCrop(cfg.img_size),
        T.ToTensor(),
        T.Normalize(mean=cfg.imagenet_mean, std=cfg.imagenet_std),
    ])
