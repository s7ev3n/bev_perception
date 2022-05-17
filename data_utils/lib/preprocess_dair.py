import imp
from .grndseg import segmentation
from .io_utils import *
import cv2
import numpy as np
import logging
from .algo_utils import *

CLASS_NAMES = ['road', 'car', 'bigcar', 'vru', 'occlusion']

CLASS_MAPPING = {
    'Car': 'car',
    'Truck': 'bigcar',
    'Van' : 'car',
    'Bus' : 'bigcar',
    'Pedestrian' : 'vru',
    'Cyclist' : 'vru',
    'Tricyclist': 'vru',
    'Motorcyclist' : 'vru',
    'Barrowlist' : 'vru'
}

CLASS_COLORS = {
    'road'      : (128,  64, 128), 
    'occlusion' : (192, 192, 192), 
    'vehicle'   : (  0,   0, 142),
    'bigcar'    : (  0,   0,  70),
    'vru'       : (220,  20,  60),
    'black'     : (  0,   0,   0),
    'white'     : (255, 255, 255)
}

def name2classid(name: str) -> int:
    """
    Desription:
        Maps a name to the index in CLASS_NAMES.
    
    Params:
        name: String, class name in DAIR dataset.
    Returns:
        index: int, index number in CLASS_NAMES. 
               If failed, return None.
    """
    if name in CLASS_MAPPING:
        return CLASS_NAMES.index(CLASS_MAPPING[name])
    else:
        return None

def ground_seg(points: np.array) -> np.array:
    """
    Descirption:
        Segment ground points from point cloud. To use the method, 
        lib/grndseg should be compiled first, follow the instrcution
        in README to install the dependency and compile it.

    Parmas:
        points: point cloud to be segmented, np.array of shape (n, 3)
    
    Returns:
        grnd_pts: ground points, np.array of shape (m, 3)
    """
    label = segmentation.segment(points)
    grnd_index = np.flatnonzero(label)
    grnd_pts = points[grnd_index]
    return grnd_pts

def get_bev_data(map_config, num_cls = None) -> np.array:
    resolution = map_config.resolution
    extents = map_config.extents

    H = (extents.top - extents.bottom) / resolution
    W = (extents.left - extents.right) / resolution
    
    if num_cls is not None:
        bev = np.zeros(shape=(num_cls, int(H), int(W)))
    else:
        bev = np.zeros(shape=(int(H), int(W)))
    return bev

def get_road_mask(grnd_pts, map_config):
    """
    Descirption:
        Read ground points and discretize into image pixel.

    Params:
        grnd_pts: ground points in camera coordinate, 
                  np.array of shape (n, 3)
        map_config: map_config file
    Returns:
        bev_data: ground pos 
    """
    bev_data = get_bev_data(map_config)
    
    uv = xy2uv(grnd_pts, map_config, 'camera')

    u, v = uv[:,0], uv[:,1]
    resolution = map_config.resolution
    for i, j in zip(u, v):
        i, j = int(i/resolution), int(j/resolution)
        bev_data[j][i] = 1
    
    # fill holes
    bev_data = fill_holes(bev_data)
    
    return bev_data

def get_occlusion_mask(obj_list, calib, map_config):
    # TODO occlusion needs tuning
    extents = map_config.extents
    resolution = map_config.resolution
    xlim = (extents.right, extents.left, resolution)
    ylim = (extents.bottom, extents.top, resolution)

    labels = []
    for obj in obj_list:
        if obj.type == 'Car':
            box3d = obj.get_box3d_corners()[:4, :]
            R, T = calib.vlidar_to_camera_extrinsic()
            box3d_cam = transform(box3d, R, T)
            labels.append(box3d_cam)
    # Origin of light source
    x0, y0 = 0, 0
    occlusion = wall_tracking(labels, x0, y0, xlim, ylim)

    return cv2.flip(occlusion.astype(np.int32), 0)


def get_object_mask(bev_data, obj_list, calib, config):
    """
    Description:
        Draw object boxes on bev_data.
    """
    for obj in obj_list:
        # NOTE in DAIR dataset, camera and lidar 3D annotation
        #      are both in lidar coordinate, so get_box3d_corners() 
        #      returns box3d corners in lidar coordinate
        box3d = obj.get_box3d_corners()[:4, :]
        R, T = calib.vlidar_to_camera_extrinsic()
        box3d_cam = transform(box3d, R, T)
        box3d_cam_bev = xy2uv(box3d_cam, \
                              config.map_config, 'camera')
        
        box3d_cam_bev = box3d_cam_bev / config.map_config.resolution
        # Render the bounding box to the appropriate mask layer
        cls_id = name2classid(obj.type)
        if cls_id is not None:
            render_polygon(bev_data[cls_id], box3d_cam_bev)

    return bev_data

def get_fov_mask(instrinsics, config):
    """
    Description:
        Filter out bev image pixel outside FOV of camera.
    """
    # Get calibration parameters
    fu, cu = instrinsics[0, 0], instrinsics[0, 2]

    # Construct a grid of image coordinates
    extents = config.map_config.extents
    resolution = config.map_config.resolution
    x = np.arange(extents.right, extents.left, resolution)
    z = np.arange(extents.bottom, extents.top, resolution)
    ucoords = x / z[:, None] * fu + cu
    # Return all points which lie within the camera bounds
    fov = (ucoords >= 0) & (ucoords < config.img.width)

    return cv2.flip(fov.astype(np.int32), 0)

