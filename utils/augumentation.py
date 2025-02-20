import torch
import torchvision.transforms.v2 as tr
from torchvision.io import read_image
from pathlib import Path

def display_tensor_augmentation(tensor_image_original, tensor_image_transformed):
    pil_image_original = tr.ToPILImage()(tensor_image_original.cpu())
    pil_image_transformed = tr.ToPILImage()(tensor_image_transformed.cpu())

    plt.figure()
    plt.subplot(121)
    plt.imshow(pil_image_original)
    plt.axis('off')
    plt.subplot(122)
    plt.imshow(pil_image_transformed)
    plt.axis('off')
    plt.show()

transform = tr.Compose([
    tr.RandomHorizontalFlip(1),
    tr.RandomRotation(30)
])

images_path = Path.home() / 'Downloads' / 'sample images' / 'Schachtabdeckung'
#find all images from image directory
image_files = images_path.glob('*')
#Iterate over images in directory 
for image_file in image_files:
    print(image_file) # mage file names
    image = read_image(str(image_file))
    transformed_image = transform({'image': image})
    display_tensor_augmentation(image, transformed_image['image'])