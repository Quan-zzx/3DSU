import torch
import numpy as np
import open3d as o3d
from tqdm import tqdm
from loguru import logger
from fast_pytorch_kmeans import KMeans
from sklearn.metrics import pairwise_distances_argmin_min


def create_object_masks(objects, poses, cam_K, num_views, top_k, image_shape):
    '''
    为每个物体选择最佳视角:
    · KMeans选择可能的最佳视角num_views个
    · 对num_views个最佳视角，生成射线，并计算物体掩码
    · 选择投影面积最大的视角
    '''
    # 使用 Open3D 的 RaycastingScene 构建一个射线投影场景，这个场景将用于从相机视角中“投射”射线，计算物体的可见性和遮罩。
    scene = o3d.t.geometry.RaycastingScene() 
    mesh = {} # 用于存储每个物体对应的三角网格（mesh）
    cam_K = cam_K.cpu().numpy()[:3, :3] # 相机内参矩阵转换为 NumPy 数组，并取前 3x3 的部分。

    # 将 3D 点云转换为三角网格并添加到 Raycasting 场景
    for i, object_ in enumerate(tqdm(objects)):              #### 遍历所有检测出来的物体
        mesh[i] = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(object_['pcd'], 0.1)                
        mesh[i] = o3d.t.geometry.TriangleMesh.from_legacy(mesh[i])                                          ###这两行代码首先根据指定的点云数据和alpha形状参数创建一个三角网格
        try:
            scene.add_triangles(mesh[i])
        except Exception as e:
            logger.critical(f"Error adding mesh for object {i}: {e}")

    for id_object, object_ in enumerate(tqdm(objects)):
        obj_poses = []                          #### 每个物体的pose
        for idx in list(object_["id"]):
            obj_poses.append(poses[idx][:3, 3].cpu().numpy().tolist())
        obj_poses = np.array(obj_poses)
        
        if len(obj_poses) < num_views:
            top_indices = range(len(list(object_["id"])))
        else:
            kmeans = KMeans(n_clusters=num_views, max_iter=500)
            kmeans.fit_predict(torch.tensor(obj_poses, device="cuda"))
            centers = kmeans.centroids.cpu().numpy()

            top_indices, _ = pairwise_distances_argmin_min(centers, obj_poses)

        pixel_area, masks, color_idx = [], [], []
        for i in top_indices:
            dataset_idx = list(object_["id"])[i]
            view_pose = poses[dataset_idx].cpu().numpy()
            rays = o3d.t.geometry.RaycastingScene.create_rays_pinhole(
                intrinsic_matrix=cam_K,
                extrinsic_matrix=np.linalg.inv(view_pose), 
                width_px=image_shape[1], 
                height_px=image_shape[0])
            ans = scene.cast_rays(rays)
            view_mask = ans['geometry_ids'].numpy() == id_object 

            masks.append([view_mask])
            pixel_area.append([view_mask.sum()])
            color_idx.append([dataset_idx])
        
        conf = np.array(pixel_area).squeeze()
        idx_most_conf = np.argsort(conf)[::-1]
        assert top_k == 1
        idx_most_conf = idx_most_conf[:top_k][0]
        object_["color_image_idx"] = color_idx[idx_most_conf][0]
        object_["mask"] = masks[idx_most_conf][0]

    return objects