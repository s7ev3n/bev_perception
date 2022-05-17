from lib import dair
from lib import preprocess_dair as preprocess
from dotmap import DotMap
from lib.io_utils import *
from lib.algo_utils import *
import random
import logging
import numpy as np
import cv2

ROOT_DIR = '/home/wangsteven/00_repos/08_bev_gt/data-dair-i/single-infrastructure-side-example'
CONFIG_P = '/home/wangsteven/00_repos/08_bev_gt/bev-dataset-devkit/config/dataset/sample_dair.yml'
INDEX = 1

def test_DAIRDataset():
    logging.info("Testing dair.py")
    dataset = dair.DAIRDataset(ROOT_DIR)
    num_data = len(dataset)
    dataframe = dataset[random.randint(0, num_data - 1)]
    logging.info("The dataset has {} samples.".format(num_data))
    return dataset

def test_get_road_mask():
    config = DotMap(read_yaml(CONFIG_P))
    bev = preprocess.get_bev_data(config.map_config)
    logging.info("Testing get_bev_data, shape {}".format(bev.shape))

    dataset = dair.DAIRDataset(ROOT_DIR)
    calib = dataset[0].calib
    R, T = calib.vlidar_to_camera_extrinsic()
    grnd_points = read_pcd(config.grnd_pcd)
    max_x = np.max(grnd_points[:, 0])
    max_y = np.max(grnd_points[:, 1])
    min_x = np.min(grnd_points[:, 0])
    min_y = np.min(grnd_points[:, 1])
    logging.info("lidar grnd points top left corner: ({x}, {y})".format(x=max_x, y=max_y))
    logging.info("lidar grnd points right bottom corner: ({x}, {y})".format(x=min_x, y=min_y))
    grnd_points_cam = filter_points(preprocess.transform(grnd_points, R, T),
                                    config.map_config.extents,
                                    'camera')
    logging.info("grnd_ppints[0] {}".format(grnd_points[0]))
    logging.info("grnd_ppints_cam[0] {}".format(grnd_points_cam[0]))
    max_x = np.max(grnd_points_cam[:, 0])
    max_z = np.max(grnd_points_cam[:, 2])
    min_x = np.min(grnd_points_cam[:, 0])
    min_z = np.min(grnd_points_cam[:, 2])
    logging.info("camera grnd points top left corner: ({x}, {z})".format(x=max_x, z=max_z))
    logging.info("lidar grnd points right bottom corner: ({x}, {z})".format(x=min_x, z=min_z))


    road_mask = preprocess.get_road_mask(grnd_points_cam, config.map_config)
    road_mask = road_mask * 255
    cv2.imwrite('road_mask.png', road_mask)

    
def test_object_mask():
    config = DotMap(read_yaml(CONFIG_P))
    bev_data = preprocess.get_bev_data(config.map_config, 
                                       len(preprocess.CLASS_NAMES))
    dataset = dair.DAIRDataset(ROOT_DIR)
    obj_list = dataset[INDEX].labels_camera
    calib = dataset[INDEX].calib
    obj_mask = preprocess.get_object_mask(bev_data, obj_list, calib, config)
    
    car = obj_mask[1] * 255
    cv2.imwrite('car_mask.png', car)
    
def test_get_fov_mask():
    config = DotMap(read_yaml(CONFIG_P))
    
    dataset = dair.DAIRDataset(ROOT_DIR)
    calib = dataset[INDEX].calib
    K, D = calib.camera_intrinsic()
    fov_mask = preprocess.get_fov_mask(K, config)
    fov = fov_mask.astype(np.int32) * 255
    cv2.imwrite('fov_mask.png', fov)


def test_get_occlusion_mask():
    config = DotMap(read_yaml(CONFIG_P))
    bev_data = preprocess.get_bev_data(config.map_config, 
                                       len(preprocess.CLASS_NAMES))
    dataset = dair.DAIRDataset(ROOT_DIR)
    obj_list = dataset[INDEX].labels_lidar
    calib = dataset[INDEX].calib
    occ_mask = preprocess.get_occlusion_mask(obj_list, calib, config.map_config)
    occ = occ_mask.astype(np.int32) * 255

    cv2.imwrite('occ_mask.png', occ)
    return occ_mask


def test_occ_refine():
    config = DotMap(read_yaml(CONFIG_P))
    dataset = dair.DAIRDataset(ROOT_DIR)
    pcd_p = dataset[INDEX].pointcloud_p
    calib = dataset[INDEX].calib
    points = read_pcd(pcd_p)
    R, T = calib.vlidar_to_camera_extrinsic()
    points_cam = transform(points, R, T)
    points_cam = filter_points(points_cam, config.map_config.extents, 'camera')
    points_uv = xy2uv(points_cam, config.map_config, 'camera')
    u, v = points_uv[:, 0], points_uv[:, 1]
    
    resolution = config.map_config.resolution
    extents = config.map_config.extents
    H = (extents.top - extents.bottom) / resolution
    W = (extents.left - extents.right) / resolution
    bev_data = np.zeros(shape=(int(H), int(W)))
    for i, j in zip(u, v):
        i, j = int(i/resolution), int(j/resolution)
        bev_data[j][i] = 255
    
    cv2.imwrite('point_frame.png', bev_data)
    bev_data = fill_holes(bev_data)
    
    occ_mask = test_get_occlusion_mask()
    bev_data_roi = bev_data * occ_mask
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(2,2))
    bev_data_roi = cv2.erode(bev_data_roi, kernel, iterations = 1)
    cv2.imwrite('point_frame_roi.png', bev_data_roi)

    bev_data_new = np.zeros(shape=(int(H), int(W)))
    H, W = bev_data.shape
    bev_data = bev_data.astype(np.int32)
    print('bev_data', np.unique(bev_data))
    print('occ_mask', np.unique(occ_mask))

    for i in range(H):
        for j in range(W):
            if bev_data[i][j] == 0 and occ_mask[i][j] == 1:
                bev_data_new[i][j] = 255
    
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
    bev_data_new = cv2.erode(bev_data_new, kernel, iterations = 1)
    cv2.imwrite('point_frame_occ.png', bev_data_new)
                

if __name__ == '__main__':
    # test_DAIRDataset()
    # test_get_road_mask()
    # test_object_mask()
    # test_get_fov_mask()
    # test_get_occlusion_mask()
    test_occ_refine()