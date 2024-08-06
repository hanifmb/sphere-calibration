import numpy as np
import open3d as o3d
import cv2


def fit_sphere(points):
    # Fit a sphere given a minimum three points
    A = np.array([[1, -2 * x, -2 * y, -2 * z] for x, y, z in points])
    B = np.array([[-x ** 2 - y ** 2 - z ** 2] for x, y, z in points])
    x = np.linalg.solve(A, B)
    x0 = x[1, 0]
    y0 = x[2, 0]
    z0 = x[3, 0]
    r = np.sqrt(x0 ** 2 + y0 ** 2 + z0 ** 2 - x[0, 0])
    return (x0, y0, z0), r


def fit_sphere_ransac(points, radius_range=(0, float("inf")), n_iterations=1000, threshold=0.02):
    best_o = None
    best_r = None
    best_inliers = []

    for _ in range(n_iterations):
        indices = np.random.choice(len(points), size=4, replace=False)
        sampled_points = points[indices]

        try:
            o, r = fit_sphere(sampled_points)
            if ~(radius_range[0] < r < radius_range[1]):
                continue  # Skip when sphere radius too small
        except np.linalg.LinAlgError:
            continue  # Skip if A is singular

        distances = np.abs(np.linalg.norm(points - np.array(o), axis=1) - r)
        inliers = np.where(distances < threshold)

        if len(inliers[0]) > len(best_inliers):
            best_o = o
            best_r = r
            best_inliers = inliers[0]
    return best_o, best_r, best_inliers


def get_pcl_o3d(points_3d, color):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_3d)
    pcd.paint_uniform_color(color)
    return pcd

def capture_img_from_o3d(vis):
    vis.poll_events()
    vis.update_renderer()
    image = vis.capture_screen_float_buffer(do_render=True)
    image_np = np.asarray(image)
    image_np = (image_np * 255).astype(np.uint8)
    image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    return image_np


if __name__ == '__main__':

    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False)  # set visible to true to show window

    radii = []
    export_imgs = []
    for idx in range(1, 9):
        # Estimate sphere center from point cloud
        ply_point_cloud = o3d.data.PLYPointCloud()
        pcd = o3d.io.read_point_cloud(f"../cloud/L1_S{idx}.xyz")
        point_cloud = np.array(pcd.points)

        l = 3  # ROI cube in meter
        point_cloud = np.array(
            [[x, y, z] for x, y, z in point_cloud if l > x > -l and l > y > 0 and l > z > -l])
        point_cloud = point_cloud[~np.all(point_cloud == [0, 0, 0], axis=1)]
        center, r, inliers = fit_sphere_ransac(point_cloud,radius_range=(0, 0.4), n_iterations=10000, threshold=0.02)
        radii.append(r)

        outliers = [x for x in range(0, len(point_cloud)) if x not in inliers]
        outlier_pcd = get_pcl_o3d(point_cloud[outliers], color=[0, 0, 1])
        inlier_pcd = get_pcl_o3d(point_cloud[inliers], color=[1, 0, 0])
        center_pcd = get_pcl_o3d([center], color=[0, 1, 0])

        vis.clear_geometries()
        vis.add_geometry(outlier_pcd)
        vis.add_geometry(inlier_pcd)
        vis.add_geometry(center_pcd)

        ctrl = vis.get_view_control()
        ctrl.camera_local_translate(3, 0, 0)
        vis.poll_events()
        vis.update_renderer()
        export_img = capture_img_from_o3d(vis)
        export_imgs.append(export_img)
        print(f"Processing pcl {idx}")

    mean_radius = np.mean(radii, axis=0)
    stacked_image = np.vstack([np.hstack(export_imgs[0:4]), np.hstack(export_imgs[4:8])])
    cv2.imwrite("../results/sphere_centers_from_pcl.jpg", stacked_image)
