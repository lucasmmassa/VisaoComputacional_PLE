from pyntcloud import PyntCloud
import numpy as np
import open3d as o3d   

cloud = o3d.io.read_point_cloud('reconstructed.ply') # Read the point cloud
o3d.visualization.draw_geometries([cloud])