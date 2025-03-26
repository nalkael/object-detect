import image_bbox_slicer as ibs


# slicing with bbox, image_bbox_slicer only works with VOC annotation now
# It would throw an error when read a non-exist value from VOC, so we need to modify the slice_helper.py in library

def data_slicer_demo(im_src, an_src, im_dst, an_dst):

    slicer = ibs.Slicer()
    slicer.keep_partial_labels = False
    slicer.ignore_empty_tiles = True
    slicer.save_before_after_map= True
    
    slicer.config_dirs(img_src=im_src, ann_src=an_src, img_dst=im_dst, ann_dst=an_dst)
    slicer.slice_by_size(tile_size=(540, 540), tile_overlap=0.3)


im_src = '/home/rdluhu/Dokumente/object_detection_project/VisDrone2019-DET/val/images'
an_src = '/home/rdluhu/Dokumente/object_detection_project/VisDrone2019-DET/val/annotations_voc'

im_dst = '/home/rdluhu/Dokumente/object_detection_project/VisDrone2019_sliced/sliced_images'
an_dst = '/home/rdluhu/Dokumente/object_detection_project/VisDrone2019_sliced/sliced_annotations'

data_slicer_demo(im_src, an_src, im_dst, an_dst)