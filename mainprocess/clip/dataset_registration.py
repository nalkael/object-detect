# dataset_registration.py
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances

# Register train and test datasets
# modify paths for custom datasets
def register_custom_datasets():
    register_coco_instances(
        "custom_train_dataset",
        {},
        "/home/rdluhu/Dokumente/object_detection_project/datasets/dataset_coco/train/_annotations.coco.json",
        "/home/rdluhu/Dokumente/object_detection_project/datasets/dataset_coco/train"
    )
    register_coco_instances(
        "custom_test_dataset",
        {},
        "/home/rdluhu/Dokumente/object_detection_project/datasets/dataset_coco/valid/_annotations.coco.json",
        "/home/rdluhu/Dokumente/object_detection_project/datasets/dataset_coco/valid"
    )

    # Set custom metadata for visualization
    MetadataCatalog.get("custom_train_dataset").set(thing_classes=['Gasschieberdeckel', 'Kanalschachtdeckel', 'Sinkkaesten', 'Unterflurhydrant', 'Versorgungsschachtdeckel', 'Wasserschieberdeckel', 'KanaldeckelQuad'])
    MetadataCatalog.get("custom_test_dataset").set(thing_classes=['Gasschieberdeckel', 'Kanalschachtdeckel', 'Sinkkaesten', 'Unterflurhydrant', 'Versorgungsschachtdeckel', 'Wasserschieberdeckel', 'KanaldeckelQuad'])
    