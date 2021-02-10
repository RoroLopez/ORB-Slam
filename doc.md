# Feature-Based SLAM for Dense Mapping

## Things to consider:

On the base of original ORB-SLAM, ORB feature points are extracted firstly, and then pose estimation is initialized by matching with points on the previous frame or relocation, keyframes are selected after the tracking of local map according to the key frame criteria.

In local mapping, points which have been added recently are removed, and then a local BA(Bundle Adjustment) is performed after creation of new map points. Unmatched ORB features in the new key frame are searched according to connected keyframes in the co-visibility graph to triangulate the new point cloud subsequently.

In loop detection, each new keyframes are tested to confirm whether a close loop is formed. If a closed loop is calculated by similarity trasnformation to fuse cloud points, and similar constrained pose(sim3) of two frames is calculated, essential graph which is a sparse subgraph in co-visibility graph is optimized to ensure the global map consistency finally.

On the basis of the above three-thread work of the original ORB-SLAM, filtering thread is added in our improved feature-based SLAM, specific are as follows:

1. Before map updating, Scharr filtering will be carried out on each existing key frame. Gradient information of the point which is used for filtering is found in the gradient graph (which has been build firstly) according to x, y coordinates of the current frame, the filter threshold names as lambdaG in the paper is set to 10000.

2. Depth information Z for points whose gradient are greater than threshold canbe found in depth map.

3. Considering the information beyond the camera (kinetic) range will have a large error, feature points which has too large depth value or ineffective will be neglected in the first optimization.

4. According to the camera model... on photo is explained.

5... on photo.

6. A statistical filter is used to remove outliers in second optimization. The filter counts distribution of the distance values of each point from N closest points, and points which are too far from its neighbors is removed. This removes isolated noise points.

7. A voxel Filters is adopted to reduce sampling finally. Due to the overlap in fields of view in multiple perspective, there will be a large number of points in the overlap area that are very similar in position. Voxel filter ensures that there is only one point in a cube of a certain size which is named as voxel, this is equivalent to a lower sampling in the three-dimensional space and saves a lot of storage space.


## Conclusions

Features are extracted by original ORB-SLAM for pose estimation. A statistical filter is used to remove isolated noise points and points whose gradient is higher than threshold are added as feature points, then a voxel Filter is adopted to reduce sampling. Maps are build by method as in ORB-SLAM finally, which is denser than that built by original ORB-SLAM. Experimental result demonstrated the imporved feature-based SLAM in this paper reconstructs scenes well with dense maps with acceptable CPU usage and efficiency.




## Possible algorithm to calculate the Rotation Matrix and Translation Vector from two images on a monocular camera

The following algorithm is used for stereo camera (2 cameras or more), so some steps may not be required in the monocular algorithm

### The algorithm
An outline:

Capture images: Itl, Itr, It+1l, It+1r
Undistort, Rectify the above images.

Compute the disparity map Dt from Itl, Itr and the map Dt+1 from It+1l, It+1r.

Use FAST algorithm to detect features in Itl, It+1l and match them.

Use the disparity maps Dt, Dt+1 to calculate the 3D posistions of the features detected in the previous steps. Two point Clouds Wt, Wt+1 will be obtained

Select a subset of points from the above point cloud such that all the matches are mutually compatible.

Estimate R,t from the inliers that were detected in the previous step.