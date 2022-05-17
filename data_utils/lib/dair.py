import numpy as np
from .io_utils import *
import os

__all__ = ['DAIRDataset']

class DAIRFrameCalib:
    def __init__(self, frame: dict, root_dir: str) -> None:
        self.frame_data = frame
        self.root_dir = root_dir
    
    def vlidar_to_camera_extrinsic(self):
        data_p = os.path.join(self.root_dir, \
                              self.frame_data['calib_virtuallidar_to_camera_path'])
        data_json = read_json(data_p)
        R = np.array(data_json['rotation']).reshape((3,3))
        T = [item[0] for item in data_json['translation']]
        T = np.array(T).reshape((3,1))
        return (R, T)

    def camera_intrinsic(self):
        data_p = os.path.join(self.root_dir, \
                              self.frame_data['calib_camera_intrinsic_path'])
        data_json = read_json(data_p)
        K = data_json['cam_K']
        K = np.array(K).reshape((3,3))
        D = data_json['cam_D'] # distortion model: plumb_bob
        D = np.array(D).reshape((len(D), ))
        return (K, D)

class ObjectInfo:
    """
    Description: 3D annoation class. Note that in DAIR dataset, camera 
                 and lidar 3D annotation are both in lidar coordinate.
    """
    def __init__(self, object_dict: dict) -> None:
        self.object_data = object_dict
        self.initialize()
    
    def initialize(self):
        self.type = self.object_data['type']
        self.alpha = int(self.object_data['alpha'])
        self.occluded_state = int(self.object_data['occluded_state'])
        self.truncated_state = int(self.object_data['truncated_state'])
        self.dimensions_3d = (float(self.object_data['3d_dimensions']['h']), \
                              float(self.object_data['3d_dimensions']['w']), \
                              float(self.object_data['3d_dimensions']['l']))
        self.location_3d   = (float(self.object_data['3d_location']['x']), \
                              float(self.object_data['3d_location']['y']), \
                              float(self.object_data['3d_location']['z']))
        self.rotation = float(self.object_data['rotation'])
        self.box_2d = self._get_box2d()

    def _get_box2d(self):
        xmin = float(self.object_data['2d_box']['xmin'])
        ymin = float(self.object_data['2d_box']['ymin'])
        xmax = float(self.object_data['2d_box']['xmax'])
        ymax = float(self.object_data['2d_box']['ymax'])
        return [(int(xmin), int(ymin)), (int(xmax), int(ymax))]

    def get_box3d_corners(self):
        x, y, z = self.location_3d
        h, w, l = self.dimensions_3d
        ry = self.rotation
        # rotation matrix
        R = np.array([[np.cos(ry), - np.sin(ry), 0],
                    [np.sin(ry),   np.cos(ry), 0],
                    [0,            0,          1]])
        # 3D bounding box corners
        x_corners = np.array([l/2, l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2])
        y_corners = np.array([w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2])
        z_corners = np.array([h/2, h/2, h/2, h/2, -h/2, -h/2, -h/2, -h/2])
        
        # corners: (3, 8)
        corners = R @ np.vstack((x_corners, y_corners, z_corners))
        corners[0,:] = corners[0,:] + x
        corners[1,:] = corners[1,:] + y
        corners[2,:] = corners[2,:] + z

        return corners.T # shape (8, 3)

class DAIRFrameData:
    """
    Desription:
        
    Params:
        data_frame: a dict consisted of the pathes of each data point, 
        looks like:
            {
                "image_path": "image/000000.jpg",
                "pointcloud_path": "velodyne/000000.pcd",
                "calib_camera_intrinsic_path": "calib/camera_intrinsic/000000.json",
                "calib_virtuallidar_to_camera_path": "calib/virtuallidar_to_camera/000000.json",
                "label_camera_std_path": "label/camera/000000.json",
                "label_lidar_std_path": "label/virtuallidar/000000.json",
                "intersection_loc": "YiZhuang11North"
            }
        root_dir: string, root_dir of the dataset
    Returns:
        dataframe: DAIRFrameData object, includes path of image and lidar raw data,
                   calibration object, label_camera_list and label_lidar_list.

    """
    def __init__(self, data_frame: dict, root_dir: str) -> None:
        self.data_frame = data_frame
        self.root_dir = root_dir
        self.initialize()

    def initialize(self):
        self.img_p = os.path.join(self.root_dir, \
                            self.data_frame['image_path'])
        self.pointcloud_p = os.path.join(self.root_dir, \
                            self.data_frame['pointcloud_path'])
        self.labels_camera = self._get_camera_annotations()
        self.labels_lidar = self._get_lidar_annotatins()
        self.intersection = self.data_frame['intersection_loc']
        self.calib = DAIRFrameCalib(self.data_frame, self.root_dir)

    def _get_camera_annotations(self) -> list:
        label_camera_p = os.path.join(self.root_dir, \
                            self.data_frame['label_camera_std_path'])
        label_camera_list = read_json(label_camera_p)
        return [ObjectInfo(obj_dict) for obj_dict in label_camera_list]

    def _get_lidar_annotatins(self) -> list:
        label_lidar_p = os.path.join(self.root_dir, \
                            self.data_frame['label_lidar_std_path'])
        label_lidar_list = read_json(label_lidar_p)
        return [ObjectInfo(obj_dict) for obj_dict in label_lidar_list]
    
class DAIRDataset(object):
    """
    Description:
        1.Given dataset root_dir, return dataset object.
        Dataset consist of dataframes, each dataframe stores:
        1)imgage_path, 2) pointcloud_p, 3)DAIRAnnotation, 4) DAIRCalib 
        2.The directory should be like below in dair dataset:
        root_dir
        ├── calib
        │   ├── camera_intrinsic
        │   └── virtuallidar_to_camera
        ├── image
        ├── label
        │   ├── camera
        │   └── virtuallidar
        ├── velodyne
        └── data_info.json
    """
    def __init__(self, root_dir) -> None:
        self.root_dir = root_dir
        self.data_info = read_json(os.path.join(self.root_dir, \
                                    'data_info.json'))

    def __getitem__(self, index):
        return DAIRFrameData(self.data_info[index], self.root_dir)

    def __len__(self):
        return len(self.data_info)

