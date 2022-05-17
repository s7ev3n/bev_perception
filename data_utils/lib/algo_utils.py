import cv2
import logging
import numpy as np
from skimage.draw import polygon

__all__ = ['xy2uv', 'transform', 'filter_points', 'wall_tracking', \
           'undistort', 'fill_holes', 'render_polygon']

def undistort(points, D):
    '''
    Params:
        points: points in camera coordinate, np.array, shape (3, n)
        D : distortion model in this case is plumb_bob, has 5 params
    Returns:
        un_points: undistorted points
    '''
    x, y, z = points[0], points[1], points[2]
    x, y = x/z, y/z
    r = np.sqrt(np.power(x, 2) + np.power(y, 2))
    k1, k2, p1, p2, k3 = D
    x_d = x * (1 + k1 * np.power(r, 2) + k2 * np.power(r, 4) + k3 * np.power(r, 6)) + \
            2 * p1 * x * y + p2 * (np.power(r, 2) + 2 * np.power(y, 2))
    y_d = y * (1 + k1 * np.power(r, 2) + k2 * np.power(r, 4) + k3 * np.power(r, 6)) + \
            2 * p2 * x * y + p1 * (np.power(r, 2) + 2 * np.power(y, 2))

    return np.vstack((x_d, y_d, np.ones_like(x_d)))

def transform(points, R, T):
    """
    Params:
        points: np.array
        R: Roation matrix R
        T: Translation matrix T
    Returns:
        points_new: np.array, transformed point
    """
    if points.shape[0] == 3:
        points_new = R @ points + T
    elif points.shape[0] > 3:
        points_new = R @ points.T + T
    else:
        logging.debug("Unknown shape: {}".format(points.shape))
        return None
    return points_new.T

def xy2uv(points, map_config, coord='lidar'):
    """
    Description:
        Transform points in camera or lidar coordinate to
        bev map image coordinate (map origin top left corner)
    
    Params:
        points: points to be transformed, np.array of shape (n, 2)
        map_config: map_config file
        coord: String, speficy which coordinate of the points.
               Please be aware lidar and camera have quite different 
               coordinate systems.
               Camera: z forward, x right, and y down
               Lidar : y forward, y left , and z up
    Returns:
        points_uv: transformed points o shape (n, 2), u and v meaning is simliar 
        to image coordinate.
    """
    if coord == 'lidar':
        x, y = points[:, 0], points[:,1]
        u = map_config.extents.left - y
        v = map_config.extents.top - x 
    elif coord == 'camera':
        x, z = points[:, 0], points[:,2]
        u = x - map_config.extents.right
        v = map_config.extents.top - z
    else:
        logging.info("Not avaliable coordinate: {}".format(coord))
    
    u, v = u[:, None], v[:, None]
    return np.hstack((u, v))

def filter_points(points, extents, coord='lidar'):
    """
    Description:
        Filter out the points outside map extents.
    
    Params:
        points: points to be filtered, np.array of shape (n, 3)
        extents: map extents
        coord: String, speficy which coordinate of the points.
               Camera: z forward, x right, and y down
               Lidar : y forward, y left , and z up 
    
    Returns:
        points_roi: points filtered, np.array of shape (m, 3)
    """
    if coord == 'lidar':
        x, y = points[:, 0], points[:, 1]
        roi_mask = np.logical_and(
            np.logical_and(extents.bottom <= x, x <= extents.top),
            np.logical_and(extents.right <= y, y <= extents.left)
        )
    elif coord == 'camera':
        x, z = points[:, 0], points[:, 2]
        roi_mask = np.logical_and(
            np.logical_and(extents.bottom <= z, z <= extents.top),
            np.logical_and(extents.right <= x, x <= extents.left)
        )
    else:
        logging.debug("Not avaliable coordinate: {}".format(coord))
        exit()
    return points[roi_mask]

def fill_holes(image, kernel_size = 3):
    """
    Description:
        The road mask image has many holes, hopefully use morphology operators
        to fill the holes.
    
    Params:
        image: image to be processed
        kernel_size: kernel size of morphology operators
    Returns:
        image: image processed
    """
    kernel = cv2.getStructuringElement(
                cv2.MORPH_RECT,(kernel_size,kernel_size))
    # cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations = 10)
    image = cv2.dilate(image, kernel, iterations = 1)
    image = cv2.erode(image, kernel, iterations = 1)

    return image

def render_polygon(mask, polygon, value = 1):
    polygon = np.ascontiguousarray(polygon).round().astype(np.int32)
    cv2.fillConvexPoly(mask, polygon, value)
    return mask

def wall_tracking(labels, x0, y0, xlim, ylim):
    """
    Description:
        Given rectangle area and boxes in it, return visibility area.
        Use it to generate object occlusion.
        Check it here: https://www.redblobgames.com/articles/visibility/
    """
    ymin, ymax, ydelta = ylim
    xmin, xmax, xdelta = xlim

    H = int(np.ceil((ymax - ymin) / ydelta))
    W = int(np.ceil((xmax - xmin) / xdelta))

    y0 = (y0 - ymin) / (ymax - ymin) * H
    x0 = (x0 - xmin) / (xmax - xmin) * W

    # NOTE: setting corner at (0,0) may lead to the ray not intersecting
    # vertices = [(0, 0), (H, 0), (H, W), (0, W)]
    # NOTE: we extend the ray such that it always intersect with something
    vertices = [(-1, -1), (H, -1), (H, W), (-1, W)]

    # initialize line segments
    segments = []
    for i in range(len(vertices)):
        src = vertices[i]
        dst = vertices[i + 1] if i < len(vertices) - 1 else vertices[0]
        segments.append((src, dst))

    # break boxes down to vertices and line segments    
    for box3d in labels:
        # get corner coordinates in top-down 2d view
        X, Y = box3d[:, 0], box3d[:, 2]

        # discretize
        Y = (Y - ymin) / (ymax - ymin) * H
        X = (X - xmin) / (xmax - xmin) * W

        for i in range(len(Y)):
            src = (Y[i], X[i])
            vertices.append(src)
            dst = (Y[i + 1], X[i + 1]) if i < len(Y) - 1 else (Y[0], X[0])
            segments.append((src, dst))

    # the angle of all rays
    thetas = np.array([np.arctan2(y - y0, x - x0) for (y, x) in vertices])

    # augmented angles
    augmented_thetas = []
    for theta in thetas:
        augmented_thetas.extend([theta - 0.00001, theta, theta + 0.00001])

    # sort augmented thetas (pi to -pi)
    order = np.argsort(augmented_thetas)[::-1]
    augmented_thetas = [augmented_thetas[idx] for idx in order]

    #
    intersections = []
    for theta in augmented_thetas:
        r_px, r_py, r_dx, r_dy = x0, y0, np.cos(theta), np.sin(theta)
        r_mag = np.sqrt(r_dx**2 + r_dy**2)

        closest_intersection = None
        closest_T1 = 10000000.0
        for (src, dst) in segments:
            s_px, s_py, s_dx, s_dy = src[1], src[0], (dst[1] - src[1]), (dst[0] - src[0])
            s_mag = np.sqrt(s_dx**2 + s_dy**2)

            # test if ray and line segment are parallel to each other
            if r_dx / r_mag == s_dx / s_mag and r_dy / r_mag == s_dy / s_mag:
                continue

            # solve the intersection equation
            T2 = (r_dx * (s_py - r_py) + r_dy * (r_px - s_px)) / (s_dx * r_dy - s_dy * r_dx)
            T1 = (s_px + s_dx * T2 - r_px) / r_dx

            # intersect behind the sensor
            if T1 < 0:
                continue

            # intersect outside the line segment
            if T2 < 0 or T2 > 1:
                continue

            # derive the coordinate of the intersection
            x, y = r_px + r_dx * T1, r_py + r_dy * T1
            if closest_intersection is None or T1 < closest_T1:
                closest_intersection = (y, x)
                closest_T1 = T1
        # this should not really happen
        if closest_intersection is not None:
            intersections.append(closest_intersection)

    # default to True
    object_shadow = np.ones((H, W), bool)
    for i in range(len(intersections)):
        y1, x1 = intersections[i]
        y2, x2 = intersections[i + 1] if i < len(intersections) - 1 else intersections[0]
        rr, cc = polygon([y0, y1, y2], [x0, x1, x2])
        I = np.logical_and(
            np.logical_and(rr >= 0, rr < H),
            np.logical_and(cc >= 0, cc < W),
        )
        object_shadow[rr[I], cc[I]] = False

    return object_shadow