# Optional list of dependencies required by the package
dependencies = ["torch", "gdown"]

from medicalnet_models.models.resnet import (
    medicalnet_resnet10,
    medicalnet_resnet10_23datasets,
    medicalnet_resnet50,
    medicalnet_resnet50_23datasets,
    medicalnet_resnet101,
    medicalnet_resnet152,
    medicalnet_resnet200
)
