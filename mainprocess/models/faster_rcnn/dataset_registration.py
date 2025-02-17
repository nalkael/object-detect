"""

run the script on the beginning of the execution

"""
from detectron2.data.datasets import register_coco_instances
from detectron2.data import MetadataCatalog, DatasetCatalog

# load config of dataset and model path
from mainprocess.models.faster_rcnn.config_loader import load_dataset_config, load_project_config

"""
# register datasets
register_coco_instances("train_dataset", {}, train_json, train_images)
register_coco_instances("valid_dataset", {}, valid_json, valid_images)
register_coco_instances("test_dataset", {}, test_json, test_images)

# Assign class names to metadata
MetadataCatalog.get("train_dataset").thing_classes = novel_classes
MetadataCatalog.get("valid_dataset").thing_classes = novel_classes
MetadataCatalog.get("test_dataset").thing_classes = novel_classes

print("Datasets registered successfully!")
"""

def register_my_dataset():

    # load the config.yaml file of the general project
    model_info = load_project_config()
    # load the dataset_config.yaml file of the Faster R-CNN model
    dataset_info = load_dataset_config(model_info["dataset_config_path"])

    # load the model_condig.yaml file of the Faster R-CNN model
    novel_classes = dataset_info["novel_classes"]
    print("Novel classes:", novel_classes)
    
    # register datasets
    register_coco_instances("train_dataset", {}, dataset_info["train_json"], dataset_info["train_images"])
    register_coco_instances("valid_dataset", {}, dataset_info["valid_json"], dataset_info["valid_images"])
    register_coco_instances("test_dataset", {}, dataset_info["test_json"], dataset_info["test_images"])

    # Assign class names to metadata
    MetadataCatalog.get("train_dataset").thing_classes = novel_classes
    MetadataCatalog.get("valid_dataset").thing_classes = novel_classes
    MetadataCatalog.get("test_dataset").thing_classes = novel_classes

    print("Datasets registered successfully!")


# call the function to register datasets
register_my_dataset()

# check if the dataset correctly registered
# print("Registered datasets: ", DatasetCatalog.list())

# datasets_dicts = DatasetCatalog.get("train_dataset")
# print(datasets_dicts[0])

# check metadata information

# metadata = MetadataCatalog.get("train_dataset")
# print("Metadata: ", metadata)
# print("Classes: ", metadata.thing_classes)