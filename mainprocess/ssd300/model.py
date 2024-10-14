import torchvision
from torchvision.models.detection.ssd import SSDClassificationHead
from torchvision.models.detection import _utils
from torchvision.models.detection import SSD300_VGG16_Weights

def create_model(num_classes=91, size=300):
    # Load the Torchvision pretrained model.
    model = torchvision.models.detection.ssd300_vgg16(weights=SSD300_VGG16_Weights.COCO_V1)

    # Retrieve the number output channels from the backbone,
    # this number then becomes the input channels for the next layer.
    # Backbone Model Output: (output_channels)
    # Next Layer Input: (in_channles)
    in_channels = _utils.retrieve_out_channels(model.backbone, (size, size))
    # print(f"in_channels: {in_channels}")
    
    # List containing number of anchors based on aspect ratios.
    num_anchors = model.anchor_generator.num_anchors_per_location()
    
    # The classification head.
    model.head.classification_head = SSDClassificationHead(
        in_channels=in_channels,
        num_anchors=num_anchors,
        num_classes=num_classes,
    )
    
    
    # Image size for transforms.
    # The format is required by the transform logic in detection models 
    # such as SSD, Faster R-CNN, etc., because of how they handle image resizing
    model.transform.min_size = (size,)
    model.transform.max_size = size
    return model

if __name__ == '__main__':
    model = create_model(8, 320)
    print(model)
    # Total parameters and trainable parameters.
    total_params = sum(p.numel() for p in model.parameters())
    print(f"{total_params:,} total parameters.")
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{total_trainable_params:,} training parameters.")