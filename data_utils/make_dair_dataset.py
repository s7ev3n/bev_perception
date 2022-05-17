import os
import lib.preprocess_dair as preprocess
from lib.io_utils import *
from dotmap import DotMap
from lib import dair
import logging

def encode_binary_labels(bev_data):
    pass

def process_frame(frame, grnd_mask, config):
    # Get bev data
    bev_data = preprocess.get_bev_data(config.map_config, \
                                        len(preprocess.CLASS_NAMES))

    # Get road mask
    bev_data[0] = grnd_mask

    # Get object mask
    obj_list = frame.labels_camera
    bev_data = preprocess.get_object_mask(bev_data, obj_list, config)

    # Get occlusion mask
    bev_data = preprocess.get_occlusion_mask(bev_data, config)

    # Get FOV mask
    bev_data = preprocess.get_fov_mask(bev_data, config)

    # Encode masks as an integer bitmask
    bev_data = encode_binary_labels(bev_data)

    # Save the result

    
def run(config_p):
    config = DotMap(read_yaml(config_p))
    # Get dair dataset
    dair_dataset = dair.DAIRDataset(config.data_dir)
    
    # Process ground points, we have only one grnd mask
    # NOTE Before using get_road_mask(), you are required 
    # to have ground pcd file beforehand.
    grnd_points = read_pcd(config.grnd_pcd)
    grnd_mask = preprocess.get_road_mask(grnd_points, config)
    logging.info("Start processing ...")
    for dataframe in dair_dataset:
        process_frame(dataframe, grnd_mask, config)


if __name__ == '__main__':
    config_p = './config/dataset/sample_dair.yml'
    run(config_p)
    