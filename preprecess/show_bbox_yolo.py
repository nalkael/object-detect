'''
 a simple program that displays an image with bounding boxes 
 based on coordinates stores in a YOLO format text file
 '''

import cv2

def draw_bbox(image, bbox_file):
    # load the image file
    img = cv2.imread(image)

    # load the txt file contains bounding box information
    with open(bbox_file, 'r') as file:
        for line in file:
            '''
            every line contains 5 values:
            class, center x, center y, width, height
            all values are relative, no longer matters if YOLO resize the image
            '''
            class_id, x_center, y_center, width, height = map(float, line.strip().split())
            '''
            delete space at the begining or at the end of every line using strip()
            split the line by spaces if there is no augument in split()
            '''
            img_height, img_width, _ = img.shape # height, width, channels
            width_min = int((x_center - width / 2) * img_width)
            width_max = int((x_center + width / 2) * img_width)
            height_min = int((y_center - height / 2) * img_height)
            height_max = int((y_center + height / 2) * img_height)
            
            # display and draw bounding box on image
            '''
            cv2.rectangle() is a function provided by the opencv library in Python
            used to draw rectangles on images
            syntax of the cv2.rectangle() function:
            cv2.rectangle(image, top-left point, bottom-right point, color, thickness)
            '''
            cv2.rectangle(img, (width_min, height_min), (width_max, height_max), (0, 8, 255), 2)
    
    # display images with bounding boxes
    cv2.imshow('Image with bounding boxes', img)
    # cv2.waitKey(0) # read an arbitary input from keyboard to interrupt
    while True:
        # wait for 1 ms to check if any key is pressed
        key = cv2.waitKey(1)

        # if a key is pressed, end the loop
        if key != -1:
            break

        if cv2.getWindowProperty('Image with bounding boxes', cv2.WND_PROP_VISIBLE) < 1:
            break

    # destroy all opencv windows
    cv2.destroyAllWindows()

# a simple example
image_file = '/home/rdluhu/Dokumente/preprecess/group-airplanes.jpg'
bbox_file = '/home/rdluhu/Dokumente/preprecess/group-airplanes.txt'

draw_bbox(image_file, bbox_file)