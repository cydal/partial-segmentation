import segmentation_models_pytorch as smp


def build_model(num_classes: int = 6, backbone: str = "resnet50", pretrained: bool = True):
    encoder_weights = "imagenet" if pretrained else None
    model = smp.DeepLabV3Plus(
        encoder_name=backbone,
        encoder_weights=encoder_weights,
        in_channels=3,
        classes=num_classes,
    )
    return model
