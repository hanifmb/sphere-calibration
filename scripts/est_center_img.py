import numpy as np
import cv2


def fit_circle(points):
    # Fit a circle using three non-collinear points
    A = np.array([[1, -2 * x, -2 * y] for x, y in points])
    B = np.array([[-x ** 2 - y ** 2] for x, y in points])
    x = np.linalg.solve(A, B)

    h = x[1, 0]
    k = x[2, 0]
    r = np.sqrt(h ** 2 + k ** 2 - x[0])[0]
    return h, k, r


def fit_circle_ransac(points, radius_range=(0, float("inf")), n_iterations=1000, threshold=2):
    best_circle = None
    best_inliers = []

    for _ in range(n_iterations):
        sample_indices = np.random.choice(len(points), 3, replace=False)
        sample_points = points[sample_indices]

        try:
            circle = fit_circle(sample_points)
            if ~(radius_range[0] < circle[2] < radius_range[1]):
                continue  # Circle radius threshold
        except:
            continue

        center = (circle[0], circle[1])
        r = circle[2]
        distances = np.abs(np.linalg.norm(points - center, axis=1) - r)
        inliers = np.where(distances < threshold)[0]
        if len(inliers) > len(best_inliers):
            best_circle = circle
            best_inliers = inliers

    return best_circle, best_inliers


def fit_ellipse(x, y):
    D = np.mat(np.vstack([x ** 2, x * y, y ** 2, x, y, np.ones(len(x))])).T
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
    params = np.asarray(a).reshape(-1)
    return params


def get_center(ep):
    a, b, c, d, e, f = ep[0], ep[1] / 2, ep[2], ep[3] / 2, ep[4] / 2, ep[5]
    num = b * b - a * c
    x0 = (c * d - b * e) / num
    y0 = (a * e - b * d) / num
    return int(y0) + 1, int(x0) + 1


def get_angle_rot(ep):
    a, b, c, d, e, f = ep[0], ep[1] / 2, ep[2], ep[3] / 2, ep[4] / 2, ep[5]

    if b == 0 and a < c:
        rot = 0
    elif b == 0 and a > c:
        rot = np.rad2deg(np.pi * 0.5)
    elif b != 0 and a < c:
        rot = np.rad2deg(0.5 * np.arctan2(2 * b, (a - c)))
    elif b != 0 and a > c:
        rot = np.rad2deg((np.pi / 2) + 0.5 * np.arctan2(2 * b, (a - c)))
    else:
        rot = np.nan
    return -rot


def get_axis_length(ep):
    a, b, c, d, e, f = ep[0], ep[1] / 2, ep[2], ep[3] / 2, ep[4] / 2, ep[5]
    up = 2 * (a * e * e + c * d * d + f * b * b - 2 * b * d * e - a * c * f)
    down1 = (b * b - a * c) * ((c - a) * np.sqrt(1 + 4 * b * b / ((a - c) * (a - c))) - (c + a))
    down2 = (b * b - a * c) * ((a - c) * np.sqrt(1 + 4 * b * b / ((a - c) * (a - c))) - (c + a))
    res1 = np.sqrt(up / down1)
    res2 = np.sqrt(up / down2)

    semi_major = max(res1, res2)
    semi_minor = min(res1, res2)

    return int(semi_major), int(semi_minor)


def draw_x(center_point, image, size=20, color=(0, 0, 255), thickness=3):
    x, y = center_point
    cv2.line(image, (x - size, y - size), (x + size, y + size), color, thickness)
    cv2.line(image, (x - size, y + size), (x + size, y - size), color, thickness)


def resize_img(image, new_width):
    if image is None:
        raise ValueError("Error: Provided image is not valid.")
    original_height, original_width = image.shape[:2]
    aspect_ratio = original_height / original_width
    new_height = int(new_width * aspect_ratio)
    resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

    return resized_image


def calc_sphere_center(xy, r):
    xy_h = np.hstack((xy, np.ones((len(xy), 1))))
    xy_h = np.sqrt(1 / np.sum(xy_h * xy_h, axis=1))[:, np.newaxis] * xy_h
    wPerCosAlpha = np.linalg.lstsq(xy_h, np.ones(len(xy)), rcond=None)[0]
    cosAlpha = 1 / np.linalg.norm(wPerCosAlpha)
    w = cosAlpha * wPerCosAlpha
    d = r / np.sqrt(1 - cosAlpha ** 2)

    S0 = d * w
    return S0


def draw_ellipse_implicit(ellipse_params_px, image, color=(0, 0, 255), thickness=4):
    center = get_center(ellipse_params_px)
    semi_major, semi_minor = get_axis_length(ellipse_params_px)
    angle = get_angle_rot(ellipse_params_px)

    # Draw the ellipse on the image
    cv2.ellipse(image, center, (semi_major, semi_minor), angle, 0, 360, color, thickness)

    return image


def get_edges(img):
    if len(img.shape) == 3 and img.shape[2] == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    gray = cv2.equalizeHist(gray)
    edges = cv2.Canny(gray, threshold1=325, threshold2=350)
    y, x = edges.nonzero()
    points_edge = np.column_stack((x, y))
    return points_edge


if __name__ == "__main__":
    result_images = []
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

        # visualizing the ellipse
        ellipse_params_px = fit_ellipse(points_edge[inliers][:, 1], points_edge[inliers][:, 0])
        draw_ellipse_implicit(ellipse_params_px, img, color=(0, 0, 255), thickness=10)

        # project the sphere center point to the image
        center_img = k @ center_3d
        center_img = center_img / center_img[2]
        center_img = np.array(center_img[:2], int)
        draw_x(center_img, img, color=(0, 0, 255), thickness=3)

        result_images.append(img)
        print(f"Processing image {idx}")

    stacked_image = np.vstack([np.hstack(result_images[0:4]), np.hstack(result_images[4:8])])
    cv2.imwrite("../results/sphere_centers_from_img.jpg", stacked_image)
