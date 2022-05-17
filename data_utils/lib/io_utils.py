import open3d as o3d
import numpy as np
import json
import cv2
import yaml

__all__ = ['read_pcd', 'save_pcd', 'read_json', 'read_yaml','read_image', 'save_image']


def read_pcd(pcd_p: str) -> np.array:
    """
    Description:
        Reads a specified pcd file using open3d.
    
    Params:
        pcd_p: String, file path of pcd file

    Returns:
        points: np.array of shape (N, 3), pcd points
    """
    return np.asarray(o3d.io.read_point_cloud(pcd_p).points)

def save_pcd(data_np: np.array, pcd_p='sample.pcd') -> bool:
    """
    Description:
        Saves to a specified pcd file using open3d.
    
    Params:
        data_np: np.array of shape (N, 3), 3d points
        pcd_p: String, full save full of pcd file

    Returns:
        success: bool, if the data is successfull saved 
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(data_np)
    return o3d.io.write_point_cloud(pcd_p, pcd)

def read_json(json_p: str):
    """
    Description:
        Reads a specified image file.
    
    Params:
        json_p: String, path of json file
    Returns:
        data: json object
    """
    with open(json_p, 'r') as f:
        data = json.load(f)
    return data


def read_yaml(yaml_p: str):
    with open(yaml_p, 'r') as f:
        return yaml.safe_load(f)

def read_image(img_p: str):
    """
    Description:
        Saves an image to a specified file.
    
    Params:
        img_p: Sting, representing the path of the image to be read.
    Returns:
        img: image data
             **Note: image data is will have the channels stored in B G R order.**
    """
    return cv2.imread(img_p)

def save_image(img_p: str, img_data):
    """
    Params:
        img_p: String 
    """
    return cv2.imwrite(img_p, img_data)