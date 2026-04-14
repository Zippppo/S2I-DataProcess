"""
虚拟相机系统 - 用于深度渲染生成皮肤点云

功能：
1. 使用PyRender渲染深度图
2. 将深度图转换为3D点云（世界坐标）

关键概念：
1. CT数据使用毫米(mm)单位
2. PyRender使用OpenGL坐标系（右手系：X右，Y上，Z向外）
3. 输出点云直接使用世界坐标（毫米），不需要归一化

复用自CT2PointCloud项目（简化版，去除归一化）

作者：rongkun
日期：2025-12
"""

import numpy as np
import trimesh
import pyrender
from typing import Tuple, Optional
import warnings


class VirtualCamera:
    """
    虚拟相机类

    坐标系说明：
    - OpenGL/PyRender: X右，Y上，Z向外（指向观察者）
    - 相机默认看向-Z方向
    """

    def __init__(self,
                 fov: float = 60.0,
                 resolution: Tuple[int, int] = (512, 512),
                 z_near: float = 100.0,
                 z_far: float = 2000.0):
        """
        初始化虚拟相机

        参数:
            fov: 视场角（度）
            resolution: 深度图分辨率 (width, height)
            z_near: 近裁剪面距离（毫米）
            z_far: 远裁剪面距离（毫米）
        """
        self.fov = fov
        self.resolution = resolution
        self.z_near = z_near
        self.z_far = z_far

        # 相机位姿（4x4变换矩阵）
        self.camera_pose = np.eye(4)

        # 创建PyRender相机对象
        self.camera = pyrender.PerspectiveCamera(
            yfov=np.radians(fov),
            aspectRatio=resolution[0] / resolution[1],
            znear=z_near,
            zfar=z_far
        )

    def look_at(self,
                eye: np.ndarray,
                center: np.ndarray,
                up: np.ndarray = np.array([0, 1, 0])):
        """
        设置相机位姿（经典的look_at方法）

        参数:
            eye: 相机位置 (3,) - 单位：毫米
            center: 相机看向的目标点 (3,) - 单位：毫米
            up: 上方向向量 (3,) - 通常是 [0, 1, 0]
        """
        forward = center - eye
        forward = forward / np.linalg.norm(forward)

        right = np.cross(forward, up)
        right = right / np.linalg.norm(right)

        up_corrected = np.cross(right, forward)

        rotation = np.eye(4)
        rotation[:3, 0] = right
        rotation[:3, 1] = up_corrected
        rotation[:3, 2] = -forward
        rotation[:3, 3] = eye

        self.camera_pose = rotation

    def get_pose(self) -> np.ndarray:
        """返回当前相机位姿矩阵"""
        return self.camera_pose.copy()

class DepthRenderer:
    """
    深度图渲染器（使用PyRender）
    """

    def __init__(self):
        self.renderer = None
        self.scene = None

    def render(self,
               mesh_data,
               camera: VirtualCamera) -> np.ndarray:
        """
        渲染深度图

        参数:
            mesh_data: MeshData对象或trimesh.Trimesh对象或列表
            camera: VirtualCamera对象

        返回:
            depth_map: (H, W) 深度图，单位：毫米
        """
        # 确保mesh_data是列表
        if not isinstance(mesh_data, list):
            mesh_data = [mesh_data]

        # 创建场景
        self.scene = pyrender.Scene()

        # 添加mesh
        for mesh in mesh_data:
            # 转换为Trimesh对象
            if isinstance(mesh, trimesh.Trimesh):
                tri_mesh = mesh
            else:
                tri_mesh = trimesh.Trimesh(
                    vertices=mesh.vertices,
                    faces=mesh.faces
                )

            # 创建双面材质
            material = pyrender.MetallicRoughnessMaterial(
                baseColorFactor=[1.0, 1.0, 1.0, 1.0],
                metallicFactor=0.0,
                roughnessFactor=1.0,
                doubleSided=True  # 关键：双面渲染
            )

            # 创建mesh
            pr_mesh = pyrender.Mesh.from_trimesh(
                tri_mesh,
                material=material,
                smooth=False
            )
            self.scene.add(pr_mesh)

        # 添加相机
        self.scene.add(camera.camera, pose=camera.get_pose())

        # 创建离屏渲染器
        self.renderer = pyrender.OffscreenRenderer(
            viewport_width=camera.resolution[0],
            viewport_height=camera.resolution[1]
        )

        try:
            # 渲染
            _, depth = self.renderer.render(self.scene)
            return depth
        finally:
            # 清理
            self.cleanup()

    def cleanup(self):
        """清理渲染器资源"""
        if self.renderer is not None:
            self.renderer.delete()
            self.renderer = None
        self.scene = None


def depth_to_pointcloud(depth_map: np.ndarray,
                        camera: VirtualCamera,
                        world_coords: bool = True) -> np.ndarray:
    """
    将深度图转换为点云

    参数:
        depth_map: (H, W) 深度图，单位：毫米
        camera: VirtualCamera对象
        world_coords: True=世界坐标系，False=相机坐标系

    返回:
        points: (N, 3) 点云，单位：毫米（世界坐标）

    坐标系说明:
        - OpenGL相机坐标系: X向右, Y向上, Z向后(指向观察者)
        - 相机看向-Z方向，所以物体在相机前方时 z < 0
    """
    H, W = depth_map.shape

    # 计算相机内参
    fx = fy = H / (2.0 * np.tan(camera.camera.yfov / 2.0))
    cx, cy = W / 2.0, H / 2.0

    # 创建像素坐标网格
    u, v = np.meshgrid(np.arange(W), np.arange(H))

    # 获取有效深度掩码
    valid_mask = depth_map > 0

    # 反投影到3D
    depth = depth_map
    x = (u - cx) * depth / fx
    y = -(v - cy) * depth / fy  # 取负：图像v向下，相机Y向上
    z = -depth  # 取负：物体在相机前方(-Z方向)

    # 组合为点云
    points = np.stack([x, y, z], axis=-1)  # (H, W, 3)
    points = points.reshape(-1, 3)  # (H*W, 3)

    # 只保留有效点
    valid = valid_mask.reshape(-1)
    points = points[valid]

    # 转换到世界坐标系
    if world_coords:
        points_homo = np.hstack([points, np.ones((len(points), 1))])
        points = (camera.get_pose() @ points_homo.T).T[:, :3]

    return points


def generate_sensor_pointcloud(
    skin_mesh: trimesh.Trimesh,
    camera_distance: float = 800.0,
    fov: float = 60.0,
    resolution: Tuple[int, int] = (512, 512),
    z_near: float = 100.0,
    z_far: float = 2000.0
) -> np.ndarray:
    """
    生成传感器点云（皮肤表面）

    参数:
        skin_mesh: 皮肤mesh
        camera_distance: 相机距离（毫米）
        fov: 视场角（度）
        resolution: 深度图分辨率
        z_near: 近裁剪面
        z_far: 远裁剪面

    返回:
        sensor_pc: (N, 3) 皮肤点云（世界坐标，毫米）
    """
    # 计算mesh中心
    mesh_center = skin_mesh.centroid

    # 创建相机
    camera = VirtualCamera(
        fov=fov,
        resolution=resolution,
        z_near=z_near,
        z_far=z_far
    )

    # 设置相机位姿（从上方看向中心）
    # CT坐标系：Y轴通常是前后方向
    eye = mesh_center + np.array([0, camera_distance, 0])
    camera.look_at(eye=eye, center=mesh_center, up=np.array([0, 0, 1]))

    # 渲染深度图
    renderer = DepthRenderer()
    depth_map = renderer.render(skin_mesh, camera)

    # 转换为点云
    sensor_pc = depth_to_pointcloud(depth_map, camera, world_coords=True)

    return sensor_pc
