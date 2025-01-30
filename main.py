# the script is to execute the main function of the program

import yaml
import os

from preprocess.tile_large_orthomosaic import tile_large_orthomosaic

if __name__ == '__main__':
    # Load the dataset yaml file
    config_yaml = yaml.safe_load(open('config.yaml', 'r'))
    # check if config.yaml is loaded
    print(config_yaml)

    sample_orthomosaic_file = config_yaml['sample_orthomosaic_file']
    output_dir = config_yaml['orthomosaic_output_dir']

    # Tile the large orthomosaic
    # tile_large_orthomosaic(sample_orthomosaic_file, output_dir)

    # TODO: do object detection here for each tile orthomosaic

    # remove the tile orthmosaic files after processing
    print('Removing the tile orthomosaic files...')
    for file in os.listdir(output_dir):
        if file.endswith('.png'):
            os.remove(os.path.join(output_dir, file))