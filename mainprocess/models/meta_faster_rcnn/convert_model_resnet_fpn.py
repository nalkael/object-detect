import os
import torch

def update_resnet1010_fpn_model(model_path: str):
    """
    Updates a model checkpoint to match the ResNet101-FPN architecture

    Args:
        model_path (str): Path to the original model checkpoint (.pth)
    
    Returns:
        str: Path to the saved updated model checkpoint
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    model = torch.load(model_path)

    for layer_ in list(model['model'].keys()):
        if 'backbone' in layer_:
            model['model'][layer_.replace('backbone', 'backbone.bottom_up')] = model['model'][layer_]
            model['model'].pop(layer_)
        if 'res5' in layer_:
            model['model'][layer_.replace('roi_heads', 'backbone.bottom_up')] = model['model'][layer_]
            model['model'].pop(layer_)

    save_path = model_path[:-4] + "_fpn.pth"
    
    torch.save(model, save_path)
    print(f"Updated model saved at: {save_path}")


if __name__ == '__main__':
    update_resnet1010_fpn_model("outputs/meta_faster_rcnn/Meta_Faster_RCNN_model_final_coco.pth")