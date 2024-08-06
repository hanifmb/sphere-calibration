from est_center_img import *
from est_center_pcl import *
import cv2
import numpy as np


def calc_pose(A, B):
    A = np.array(A, copy=False)
    B = np.array(B, copy=False)
    assert A.shape == B.shape, "Point sets must have the same shape"
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    A_centered = A - centroid_A
    B_centered = B - centroid_B
    H = np.dot(B_centered.T, A_centered)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = np.dot(Vt.T, U.T)
    T = centroid_A - np.dot(R, centroid_B)
    return R, T


if __name__ == "__main__":

    radii = []
    center_3d_img_acc = []
    for idx in range(1, 9):
        # Estimate sphere center from point cloud
        ply_point_cloud = o3d.data.PLYPointCloud()
        pcd = o3d.io.read_point_cloud(f"../cloud/L1_S{idx}.xyz")
        point_cloud = np.array(pcd.points)

        l = 3  # ROI cube in meter
        point_cloud = np.array(
            [[x, y, z] for x, y, z in point_cloud if l > x > -l and l > y > 0 and l > z > -l])
        point_cloud = point_cloud[~np.all(point_cloud == [0, 0, 0], axis=1)]
        center, r, inliers = fit_sphere_ransac(point_cloud, radius_range=(0, 0.4), n_iterations=10000, threshold=0.02)
        center_3d_img_acc.append(center)
        radii.append(r)
    mean_radius = np.mean(radii, axis=0)
    print(f"mean radius: {mean_radius}")
    print(f"centers from pointclouds: {center_3d_img_acc}")

    center_3d_pcl_acc = []
    for idx in range(1, 9):
        filename = f"../images/C1_S{idx}.bmp"
        img = cv2.imread(filename, cv2.IMREAD_COLOR)
        points_edge = get_edges(img)

        # intrinsic matrix
        k = np.array([[4531.3, 0, 658.86],
                      [0, 4528.8, 619.01],
                      [0, 0, 1]])

        # fit an ellipse to the best circle found
        circle, inliers = fit_circle_ransac(points=points_edge, radius_range=(50, 275), n_iterations=10000,
                                                threshold=2)
        inliers_img_h = [[x, y, 1] for x, y in points_edge[inliers]]
        inliers_cam_h = inliers_img_h @ np.linalg.inv(k).transpose()  # normalize to the camera coord system
        inliers_cam_xy = [list(pair) for pair in zip(inliers_cam_h[:, 0], inliers_cam_h[:, 1])]
        center_3d = calc_sphere_center(inliers_cam_xy, r=0.3)
        center_3d_pcl_acc.append(tuple(center_3d))
    print(f"centers from images: {center_3d_pcl_acc}")

    # get the rotation and translation
    rotation, translation = calc_pose(center_3d_img_acc, center_3d_pcl_acc)
    print(f"rotation:\n{rotation}\ntranslation:\n{translation}")

    # visualization after the lidar points are transformed to the camera
    '''
    transform_mat = np.eye(4, 4)
    transform_mat[:3, :3] = rotation
    transform_mat[:3, 3] = translation
    transformed_centers_pcl = np.array([[x, y, z, 1] for x, y, z in center_3d_pcl_acc]) @ transform_mat.transpose()
    
    centers_img_o3d = get_pcl_o3d(center_3d_img_acc, color=(1, 0, 0))
    centers_pcl_o3d = get_pcl_o3d(transformed_centers_pcl[:, :3], color=(0, 0, 1))
    o3d.visualization.draw_geometries([centers_img_o3d, centers_pcl_o3d])
    '''
