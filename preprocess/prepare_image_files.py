import sqlite3
import os
from pathlib import Path

# the path of project folder
current_folder = os.path.dirname(os.path.abspath(__file__))
parent_folder = os.path.dirname(current_folder)
images_dir = os.path.join(parent_folder, 'orthomosaic')
tiles_dir = os.path.join(parent_folder, 'tiles')
# print(images_dir)
# print(tiles_dir)


# file_path = os.path.join(file_dir, 'images', 'example.txt')
# print(file_path)

def create_image_table():
    create_table_statement = '''
        CREATE TABLE IF NOT EXISTS orthomosaics (
            image_id INTEGER PRIMARY KEY,
            image_path TEXT NOT NULL,
            image_name TEXT NOT NULL
        );
        '''

    # create a database connection
    try:
        with sqlite3.connect('orthomosaics.db') as conn:
            cursor = conn.cursor()
            cursor.execute(create_table_statement)
            conn.commit()
    except sqlite3.Error as e:
        print(e)


def insert_images():
    sql_statement = """
        INSERT INTO orthomosaics (image_path, image_name)
        VALUES (?, ?)
        """
    try:
        with sqlite3.connect('orthomosaics.db') as conn:
            cursor = conn.cursor()
            cursor.execute(sql_statement, images_dir, '20220203_FR_Wirthstrasse_transparent_mosaic_group1.tif')
            cursor.execute(sql_statement, images_dir, '20221014_BN_Betriebsgelände_OG_transparent_mosaic_group1.tif')
            cursor.execute(sql_statement, images_dir, '20221027_FR_Habsburger_Str_transparent_mosaic_group1.tif')
            cursor.execute(sql_statement, images_dir, '20221123_Fehrenbachallee_transparent_mosaic_group1.tif')
            cursor.execute(sql_statement, images_dir, '20230221_GEF_Merzhausen_transparent_mosaic_group1.tif')
            cursor.execute(sql_statement, images_dir, '20230405_FR_Merzhauser_Str_PeterThumb_Str_transparent_mosaic_group1.tif')
            cursor.execute(sql_statement, images_dir, '20230717_FR_Sundgauallee_Rev2_transparent_mosaic_group1.tif')
            cursor.execute(sql_statement, images_dir, '20230808_FR_Merianstr_Rheinstr_transparent_mosaic_group1.tif')
            cursor.execute(sql_statement, images_dir, '20231116_FR_Besaconallee_B3_transparent_mosaic_group1.tif')
            cursor.execute(sql_statement, images_dir, '20240228_FR_Mathias-Blank_Str_transparent_mosaic_group1.tif')
    except sqlite3.Error as e:
        print(e)


def read_images():
    sql_statement = """
    """


if __name__ == '__main__':
    create_image_table()
    insert_images()
