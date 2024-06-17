# Roof Surface Reconstruction

This repo contains a solution for the S23DR challenge, which focuses on extracting precise 3D wireframes from 2D images captured in urban scenes. The proposed method is an improvement over the provided baseline submission and implements a naive approach more intelligently.

## Approach

The solution employs the following key techniques:

1. **Vertex Detection**: Vertex masks are extracted from the "gestalt" segmentation using dilations, erosions, and connected component analysis.
2. **Line Detection**: Line masks are preprocessed similarly to vertex detection, followed by Hough line detection to find line ends.
3. **Missing Vertex Detection**: Missing vertices are detected by checking the proximity of line ends and inferring new vertices if necessary.
4. **Line Fusion**: Detected lines are mapped onto a predefined wireframe structure by verifying their alignment with fixed lines within an angular threshold.
5. **Depth Estimation**: Depth estimation is performed by projecting a denoised point cloud onto the 2D camera plane and using a nearest neighbor interpolation technique to estimate vertex depths from neighboring points.

## Results

On the first 1000 points of the dataset, the solution achieved a mean Wireframe Editing Distance (WED) score of 1.95, showing significant improvement over the baseline submission.

## Areas of Improvement

The current solution's weakness lies in accurate depth estimation. A more robust algorithm, such as training a Fully Convolutional Neural Network (FCNN) to predict complete depth maps from sparse depth maps and scaled monocular depth maps, could potentially improve the performance further.

## Usage

Usage n everything is very well shown in the provided python notebook.

To run the solution, use the predict function on the binary form of dataset only. Without converting to the human readable form.

For prediction with the best parameters:

```python
from baseline import *
key, vertices, edges = predict(sample, visualize=False,
                                                point_radius=30,
                                                max_angle=5,
                                                extend=30,
                                                merge_th=80.0,
                                                min_missing_distance=1000.0,
                                                scale_estimation_coefficient=2.54,
                                                clustering_eps=150,
                                                interpolation_radius=10000,
                                                point_radius_scale=1,
                                                # dist_coeff=0,
                                                pointcloud_depth_coeff=1,
                                                )
```