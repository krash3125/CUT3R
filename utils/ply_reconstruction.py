import os
import time
import numpy as np
import open3d as o3d


def create_point_cloud(pts3ds_other, colors, conf, cam_dict, output_dir):
    # Process outputs for visualization.
    print("Preparing output for saving...")
    start_time = time.time()

    # Convert tensors to numpy arrays for saving.
    pts_all = []
    colors_all = []
    for pts, clr, mask in zip(pts3ds_other, colors, conf):
        pts_np = pts.cpu().numpy().reshape(-1, 3)
        clr_np = clr.cpu().numpy().reshape(-1, 3)
        mask_np = mask.cpu().numpy().reshape(-1)
        valid = np.isfinite(pts_np).all(axis=1) & (mask_np > 0.5)
        pts_all.append(pts_np[valid])
        colors_all.append(clr_np[valid])

    pts_all = np.concatenate(pts_all, axis=0)
    colors_all = np.concatenate(colors_all, axis=0)

    # Save as PLY using Open3D
    print(f"Loading point cloud with {pts_all.shape[0]} points.")
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts_all)
    pcd.colors = o3d.utility.Vector3dVector(colors_all)
    pcd_down = pcd.voxel_down_sample(voxel_size=0.02)
    print(f"Downsampled to {len(pcd_down.points)} points. Saving point cloud...")
    o3d.io.write_point_cloud(os.path.join(output_dir, "fused.ply"), pcd_down)

    total_time = time.time() - start_time
    print(f"Point cloud completed + saved in {total_time:.2f} seconds.")
    return pcd_down


def create_mesh(pcd_down, output_dir):
    # Estimate normals (needed for meshing)
    print("Calculating for mesh.")
    start_time = time.time()
    pcd_down.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
    )
    pcd_down.orient_normals_consistent_tangent_plane(k=10)

    # Poisson mesh reconstruction
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd_down, depth=9
    )
    mesh.compute_vertex_normals()

    # Crop to remove low-density artifacts (optional)
    bbox = pcd_down.get_axis_aligned_bounding_box()
    mesh_crop = mesh.crop(bbox)

    # Save mesh
    o3d.io.write_triangle_mesh(os.path.join(output_dir, "fused_mesh.ply"), mesh_crop)
    total_time = time.time() - start_time
    print(f"Mesh completed + saved in {total_time:.2f} seconds.")
    return mesh_crop
