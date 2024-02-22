import cv2
import numpy as np
import open3d as o3d

def fit_sphere(points):
    # Fit a sphere given a minimum three points
    A = np.array([[1, -2*x, -2*y, -2*z] for x, y, z in points])
    B = np.array([[-x**2-y**2-z**2] for x, y, z in points])
    x = np.linalg.solve(A, B)
    x0 = x[1, 0]
    y0 = x[2, 0]
    z0 = x[3, 0]
    r = np.sqrt(x0**2 + y0**2 + z0**2 - x[0, 0])
    return (x0, y0, z0), r

def fit_sphere_ransac(points, n_iterations, threshold):
    best_o = None
    best_r = None
    best_inliers = []

    for _ in range(n_iterations):
        indices = np.random.choice(len(points), size=4, replace=False)
        sampled_points = points[indices]

        try:
            o, r = fit_sphere(sampled_points)
            if r > 0.4: continue # Skip when sphere radius too small
        except np.linalg.LinAlgError: continue # Skip if A is singular

        distances = np.abs(np.linalg.norm(points - np.array(o), axis=1) - r)
        inliers = np.where(distances < threshold)

        if len(inliers[0]) > len(best_inliers):
            best_o = o
            best_r = r
            best_inliers = inliers[0]
    return best_o, best_r, best_inliers

def fit_circle(points):
    # Fit a circle using three non-collinear points
    A = np.array([[1, -2*x, -2*y] for x, y in points])
    B = np.array([[-x**2-y**2] for x, y in points])
    x = np.linalg.solve(A, B)

    h = x[1, 0]
    k = x[2, 0]
    r = np.sqrt(h**2+k**2-x[0])[0]
    return h, k, r

def fit_circle_ransac(points, n_iterations=10000, threshold=2):
    best_circle = None
    best_inliers = []

    for _ in range(n_iterations):
        sample_indices = np.random.choice(len(points), 3, replace=False)
        sample_points = points[sample_indices]

        try:
            circle = fit_circle(sample_points)
            if circle[2] > 300: continue # Circle radius threshold
        except: continue

        center = (circle[0], circle[1])
        r = circle[2]
        distances = np.abs(np.linalg.norm(points - center, axis=1) - r) 
        inliers = points[distances < threshold]
        if len(inliers) > len(best_inliers):
            best_circle = circle
            best_inliers = inliers

    return best_circle, best_inliers


def fit_ellipse(x, y):
    D = np.mat(np.vstack([x**2, x*y, y**2, x, y, np.ones(len(x))])).T
    S = np.dot(D.T, D)
    C = np.zeros((6, 6))
    C[0, 2] = 2
    C[1, 1] = -1
    C[2, 0] = 2
    Z = np.dot(np.linalg.inv(S), C)
    eigen_val, eigen_vec = np.linalg.eig(Z)
    eigen_val = eigen_val.reshape(1, -1)
    pos_r, pos_c = np.where(eigen_val > 0 & ~np.isinf(eigen_val))
    a = eigen_vec[:, pos_c]
    return a

def ellipse_center(a):
    a = a.reshape(-1, 1)
    b, c, d, f, g, a = a[1]/2, a[2], a[3]/2, a[4]/2, a[5], a[0]
    num = b*b - a*c
    x0=(c*d-b*f)/num
    y0=(a*f-b*d)/num
    return (int(y0[0, 0])+1, int(x0[0, 0])+1)

def ellipse_angle_of_rotation(a):
    a = a.reshape(-1, 1)
    b,c,d,e,f,a = a[1]/2, a[2], a[3]/2, a[4]/2, a[5], a[0]

    if b == 0 and a < c:
        rot = 0
    elif b == 0 and a > c:
        rot = np.rad2deg(np.pi * 0.5)
    elif b != 0 and a < c:
        rot = np.rad2deg(0.5 * np.arctan2(2*b, (a-c)))
    elif b != 0 and a > c:
        rot = np.rad2deg((np.pi / 2) + 0.5 * np.arctan2(2*b, (a-c)))
    return -rot

def ellipse_axis_length(a):
    a = a.reshape(-1, 1)
    b, c, d, f, g, a = a[1]/2, a[2], a[3]/2, a[4]/2, a[5], a[0]
    up = 2*(a*f*f+c*d*d+g*b*b-2*b*d*f-a*c*g)
    down1=(b*b-a*c)*( (c-a)*np.sqrt(1+4*b*b/((a-c)*(a-c)))-(c+a))
    down2=(b*b-a*c)*( (a-c)*np.sqrt(1+4*b*b/((a-c)*(a-c)))-(c+a))
    res1=np.sqrt(up/down1)
    res2=np.sqrt(up/down2)
    return (int(res1[0,0]), int(res2[0, 0]))

def draw_x(image, center_point, size=20, color=(0, 0, 255), thickness=3):
    x, y = center_point
    cv2.line(image, (x - size, y - size), (x + size, y + size), color, thickness)
    cv2.line(image, (x - size, y + size), (x + size, y - size), color, thickness)

def calc_sphere_center(XY, r):

    XY_h = np.hstack((XY, np.ones((len(XY), 1))))
    XY_h = np.sqrt(1 / np.sum(XY_h * XY_h, axis=1))[:, np.newaxis] * XY_h
    wPerCosAlpha = np.linalg.lstsq(XY_h, np.ones(len(XY)), rcond=None)[0]
    cosAlpha = 1 / np.linalg.norm(wPerCosAlpha)
    w = cosAlpha * wPerCosAlpha
    d = r / np.sqrt(1 - cosAlpha**2)

    S0 = d * w
    return S0

if __name__ == "__main__":
    file_path = 'ids.txt'
    lines = []
    with open(file_path, 'r') as file:
        for line in file:
            lines.append(line.strip())

    acc_center_img = []
    acc_center_pcd = []
    for record_id in lines:
        img = cv2.imread(f"images/Dev0_Image_w1920_h1200_{record_id}.jpg")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (21, 21), 0)
        edges = cv2.Canny(blurred, threshold1=100, threshold2=200)
        y, x = edges.nonzero()
        points_edge = np.column_stack((x, y))

        # Fit ellipse to the image (visualization purpose)
        circle, inliers = fit_circle_ransac(points_edge)
        x_inlier_px = inliers[:, 0]
        y_inlier_px = inliers[:, 1]
        a_ellipse_px = fit_ellipse(y_inlier_px, x_inlier_px)

        center = ellipse_center(a_ellipse_px)
        axis = ellipse_axis_length(a_ellipse_px)
        angle = ellipse_angle_of_rotation(a_ellipse_px)
        cv2.ellipse(img, center, axis, angle[0, 0], 0, 360, (0, 255, 0), 2)

        # Calculate sphere center from ellipse inlier points
        inlier_homogenous = np.array([[x, y, 1] for x, y in inliers])
        K = np.array([[1250, 0, 960],
                      [0, 1250, 600],
                      [0, 0, 1]])
        inlier_cam = inlier_homogenous @ np.linalg.inv(K).T
        x_inlier_cam = inlier_cam[:, 0]
        y_inlier_cam = inlier_cam[:, 1]

        r = 300 # Sphere radius in mm
        sphere_center = calc_sphere_center(np.column_stack((x_inlier_cam, y_inlier_cam)), r)
        center_pts = K @ sphere_center
        center_pts = center_pts / center_pts[2]
        draw_x(img, np.array(center_pts[:2], int))

        # Estimate sphere center from point cloud
        ply_point_cloud = o3d.data.PLYPointCloud()
        pcd = o3d.io.read_point_cloud(f"cloud/test_{record_id}.xyz")
        point_cloud = np.array(pcd.points)

        l = 3 # ROI cube in meter
        point_cloud = np.array([[x, y, z] for x, y, z in point_cloud if x < l and x > -l and y < l and y > 0 and z < l and z > -l])
        point_cloud = point_cloud[~np.all(point_cloud == [0, 0, 0], axis=1)]
        center, r, inliers = fit_sphere_ransac(point_cloud, 10000, 0.02)
        center_mm = np.array(center) * 1000

        cv2.namedWindow('img', cv2.WINDOW_NORMAL)
        cv2.imshow('img', img)

        key = cv2.waitKey(0) & 0xFF
        if key == ord('q'): # Select valid ellipse manually
            acc_center_img.append(sphere_center)
            acc_center_pcd.append(center_mm)
        elif key == ord('w'): continue # Skip ellipse
        elif key == ord('e'): break # Finish

    cv2.destroyAllWindows()

    # Calculate the rotation r from p to q pointcloud
    p = np.array(acc_center_img)
    q = np.array(acc_center_pcd)

    centroid_p = np.mean(p, axis=0)
    centroid_q = np.mean(q, axis=0)

    p_centered = p - centroid_p
    q_centered = q - centroid_q

    pq = p_centered.T @ q_centered
    U, S, Vt = np.linalg.svd(pq)
    r = Vt.T @ U.T

    if np.linalg.det(r) < 0:
        V_altered = np.copy(Vt.T)
        V_altered[:, 2] = V_altered[:, 2] * -1
        r = V_altered @ U.T

    # Calculate the translation from p to q
    t =  centroid_q - r @ centroid_p

    # Calculate the scale from p to q pointcloud
    p_rotated = p_centered @ r.T
    nomi = np.sum(np.sum(p_rotated * q, axis=1))
    deno = np.sum(np.sum(p_rotated * p_rotated, axis=1))
    scale = nomi / deno

    # Visualization
    print(f"rotation: {r}")
    print(f"scale: {scale}")

    center_pts_img = o3d.geometry.PointCloud()
    center_pts_img.points = o3d.utility.Vector3dVector(p_rotated * scale)
    center_pts_img.paint_uniform_color([1, 0, 0])

    center_pts_pcd = o3d.geometry.PointCloud()
    center_pts_pcd.points = o3d.utility.Vector3dVector(q_centered)
    center_pts_pcd.paint_uniform_color([0, 0, 1])

    o3d.visualization.draw_geometries([center_pts_img, center_pts_pcd])
