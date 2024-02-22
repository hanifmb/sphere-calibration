# sphere-calibration
Camera-lidar calibration is carried out by calculating the center of the sphere both from lidar point cloud and camera images. The pose can then be retrieved by performing optimal point registration. A minimum of three sphere center point correspondences are required to calculate the extrinsic parameter.

Ellipse robust fitting is based on circle fit with RANSAC followed by ellipse fitting from the circle inlier points as described in [Automatic LiDAR-Camera Calibration of Extrinsic Parameters Using a Spherical Target
](https://ieeexplore.ieee.org/document/9197316). Sphere centers from images are estimated by calculating the cone parameters of the sphere on a calibrated camera ([A Minimal Solution for Image-Based Sphere Estimation
](https://doi.org/10.1007/s11263-023-01766-1)).

## References
[Visualizing an ellipse (Wolfram)](https://mathworld.wolfram.com/Ellipse.html) </br>
[Optimal point registration without scaling (nghiaho.com) ](https://nghiaho.com/?page_id=671)
