"""

run the script on the beginning of the execution

"""
from detectron2.data.datasets import register_coco_instances
from detectron2.data import MetadataCatalog, DatasetCatalog


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

def register_my_dataset(
    datasets=['train', 'valid', 'test'],
    # class names should be extented 
    class_names=[
        'gas_schieberdeckel',
        "kanal_deckel_quad", 
        "kanal_regenwassereinlass", 
        "kanal_schachtdeckel", 
        "versorgungs_deckel_eisen", 
        "versorgungs_schachtdeckel", 
        "wasser_schieberdeckel", 
        "wasser_unterflur_hydrant"
        ]
    ):
    
    for dataset in datasets:
        if dataset not in DatasetCatalog.list():
            register_coco_instances(
                f"{dataset}_dataset",
                {},
                f"datasets/dataset_coco/{dataset}/_annotations.coco.json", # json folders
                f"datasets/dataset_coco/{dataset}" # image folders
            )
        
        # assign class names to metadata
        MetadataCatalog.get(f"{dataset}_dataset").thing_classes = class_names
        print(f"Register dataset {dataset}...")


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