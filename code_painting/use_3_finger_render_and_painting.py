#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
机器人渲染器 - 修复相机朝向问题 + 视频生成功能
"""

import os
import numpy as np
import sapien.core as sapien
from sapien.render import clear_cache as sapien_clear_cache
from sapien.utils.viewer import Viewer
from typing import Union, Dict, Any
import cv2
import transforms3d.euler as t3d_euler
import transforms3d.quaternions as t3d_quat
from envs.robot import Robot
import json


class RobotRenderer:
    """机器人渲染器类，用于根据末端位置和相机外参渲染机器人场景"""
    
    def __init__(self, 
                 image_width: int = 640, 
                 image_height: int = 360,
                 enable_viewer: bool = False,
                 fovy_deg: float = 90.0,
                 ground_height: float = 0.0,
                 world_z_offset: float = 0.0,
                 arms_z_offset: float = 0.0,
                 debug_minimal_alignment: bool = False,
                 debug_zero_rotation: bool = False,
                 ):
        """
        初始化机器人渲染器
        """
        self.image_width = image_width
        self.image_height = image_height
        self.enable_viewer = enable_viewer
        # 视场角做合理范围限制，避免极端值导致渲染异常
        self.fovy_deg = float(np.clip(fovy_deg, 30.0, 120.0))
        # 地面高度与世界Z偏移
        self.ground_height = float(ground_height)
        self.world_z_offset = float(world_z_offset)
        self.arms_z_offset = float(arms_z_offset)
        self.debug_minimal_alignment = bool(debug_minimal_alignment)
        self.debug_zero_rotation = bool(debug_zero_rotation)
        
        # 初始化SAPIEN环境
        self._setup_sapien_scene()
        self._load_robot()
        self._setup_camera()
        
        print("Robot Renderer initialized successfully!")
    
    def _setup_sapien_scene(self):
        """设置SAPIEN场景"""
        # 创建引擎和渲染器
        self.engine = sapien.Engine()
        
        from sapien.render import set_global_config
        set_global_config(max_num_materials=50000, max_num_textures=50000)
        
        self.renderer = sapien.SapienRenderer()
        self.engine.set_renderer(self.renderer)
        
        # 设置光线追踪参数
        sapien.render.set_camera_shader_dir("rt")
        sapien.render.set_ray_tracing_samples_per_pixel(32)
        sapien.render.set_ray_tracing_path_depth(8)
        sapien.render.set_ray_tracing_denoiser("oidn")
        
        # 创建场景
        scene_config = sapien.SceneConfig()
        self.scene = self.engine.create_scene(scene_config)
        self.scene.set_timestep(1/250)
        
        # 添加地面（可配置高度）
        self.scene.add_ground(self.ground_height)
        
        # 设置物理材料
        self.scene.default_physical_material = self.scene.create_physical_material(
            static_friction=0.5,
            dynamic_friction=0.5,
            restitution=0
        )
        
        # 设置环境光
        self.scene.set_ambient_light([0.5, 0.5, 0.5])
        
        # 添加方向光
        self.scene.add_directional_light([0, 0.5, -1], [0.5, 0.5, 0.5], shadow=True)
        
        # 添加点光源
        self.scene.add_point_light([1, 0, 1.8], [1, 1, 1], shadow=True)
        self.scene.add_point_light([-1, 0, 1.8], [1, 1, 1], shadow=True)
        
        # 如果启用viewer
        if self.enable_viewer:
            self.viewer = Viewer(self.renderer)
            self.viewer.set_scene(self.scene)
            self.viewer.set_camera_xyz(x=0.4, y=0.22, z=1.5)
            self.viewer.set_camera_rpy(r=0, p=-0.8, y=2.45)
    
    def _load_robot(self):
        """加载机器人模型"""
        try:
            robot_urdf_path = "/home/pine/RoboTwin2/assets/embodiments/aloha-agilex/urdf/arx5_description_isaac.urdf"
            
            if os.path.exists(robot_urdf_path):
                with open("/home/pine/RoboTwin2/robot_config.json", "r") as f:
                    robot_cfg = json.load(f)
                    
                self.need_topp = True
                self.robot = Robot(self.scene, self.need_topp, **robot_cfg)
                self.robot.init_joints()
                print("Robot loaded successfully with dual-arm configuration")
                self.robot.print_info()
            else:
                print(f"!!! URDF file not found: {robot_urdf_path}. Creating simplified robot model...")
                
        except Exception as e:
            print(f"Failed to load robot URDF: {e}")
            print("Creating simplified robot model...")
        
        # 初始化关节状态
        self._init_joint_states()
    
    def _init_joint_states(self):
        """初始化关节状态"""
        if self.robot is not None:
            try:
                # 移动到家位置
                self.robot.move_to_homestate()
                
                # 设置初始夹爪状态 (打开)
                self.robot.left_gripper_val = 0.8
                self.robot.right_gripper_val = 0.8
                
                # 更新一步物理仿真以应用关节状态
                self.scene.step()
                
            except Exception as e:
                print(f"Failed to initialize joint states: {e}")
    
    def _setup_camera(self):
        """设置相机"""
        # 创建相机
        self.camera = self.scene.add_camera(
            name="head_camera",
            width=self.image_width,
            height=self.image_height,
            fovy=np.deg2rad(self.fovy_deg),
            near=0.01,
            far=100.0
        )
        
        # 设置默认相机位置
        self.camera.set_entity_pose(sapien.Pose([0, 0, 1.6], [0, 0, 0, 1]))

    def _opencv_to_sapien_pose(self, opencv_position: np.ndarray, opencv_rotation: np.ndarray) -> sapien.Pose:
        """
        将OpenCV相机外参转换为SAPIEN相机姿态
        
        OpenCV相机坐标系：
        - X轴：右
        - Y轴：下  
        - Z轴：前（朝向场景）
        
        SAPIEN相机坐标系：
        - X轴：右
        - Y轴：上
        - Z轴：后（从场景朝向相机）
        
        Args:
            opencv_position: OpenCV相机位置
            opencv_rotation: OpenCV相机旋转矩阵
            
        Returns:
            sapien.Pose: SAPIEN相机姿态
        """
        # OpenCV到SAPIEN的坐标系转换矩阵
        # 需要绕X轴旋转180度：Y轴翻转（下->上），Z轴翻转（前->后）
        # opencv_to_sapien = np.array([
        #     [1,  0,  0],  # X轴保持不变
        #     [0, -1,  0],  # Y轴翻转：下 -> 上
        #     [0,  0, -1]   # Z轴翻转：前 -> 后
        # ])
        opencv_to_sapien = np.array([
            [1,  0,  0],  # X轴保持不变
            [0,  1,  0],  # Y轴翻转：下 -> 上 (简单朝左偏了)
            [0,  0,  1]   # Z轴翻转：前 -> 后
        ])        
        # 转换旋转矩阵
        sapien_rotation = opencv_rotation @ opencv_to_sapien
        
        # 位置保持不变（都是世界坐标系）
        sapien_position = opencv_position.copy()
        
        # 转换为四元数
        sapien_quat = t3d_quat.mat2quat(sapien_rotation)  # [w, x, y, z]
        sapien_quat_xyzw = [sapien_quat[1], sapien_quat[2], sapien_quat[3], sapien_quat[0]]  # [x, y, z, w]
        
        return sapien.Pose(sapien_position, sapien_quat_xyzw)

    def set_robot_base_pose(self, camera_extrinsics: Dict[str, Any]):
        """
        根据相机外参设置机器人基座位置
        """
        cam_pos = np.array(camera_extrinsics['position'])
        print("1. cam_pos ==========", cam_pos)

        # 提取旋转信息
        if 'rotation' in camera_extrinsics:
            rotation_matrix = np.array(camera_extrinsics['rotation'])
        elif 'quaternion' in camera_extrinsics:
            head_quat_raw = np.array(camera_extrinsics['quaternion'])
            rotation_matrix = t3d_quat.quat2mat(head_quat_raw)
        elif 'euler' in camera_extrinsics:
            euler = camera_extrinsics['euler']
            rotation_matrix = t3d_euler.euler2mat(*euler)
        else:
            rotation_matrix = np.eye(3)
        print("2. rotation_matrix =============", rotation_matrix)
        
        # 使用OpenCV约定计算旋转角度
        if self.debug_zero_rotation:
            yaw = np.pi / 2
            print("[DEBUG] zero-rotation mode: yaw=π/2")
        else:
            # 从旋转矩阵中提取欧拉角
            cam_roll, cam_pitch, cam_yaw = t3d_euler.mat2euler(rotation_matrix, axes='sxyz')
            cam_angles_deg = np.rad2deg([cam_roll, cam_pitch, cam_yaw])
            
            # 使用相机的roll角作为机器人的yaw旋转
            yaw = cam_roll
            
            print(f"相机欧拉角(roll, pitch, yaw): {cam_angles_deg[0]:.2f}, {cam_angles_deg[1]:.2f}, {cam_angles_deg[2]:.2f} 度")
            print(f"底座偏转角度(yaw): {np.rad2deg(yaw):.2f} 度")
        
        # 创建机器人基座四元数（绕X轴旋转）
        base_quat = t3d_euler.euler2quat(yaw, 0, 0, 'sxyz')
        print(f"Base quaternion: {base_quat}")
        
        # 计算机器人基座位置（直接使用相机位置的XY，设置Z为地面高度）
        robot_base_pos = np.array([cam_pos[0], cam_pos[1], self.ground_height])
        
        # 保存基座位置供后续使用
        self._robot_base_pos = robot_base_pos.copy()
        print(f"Saved robot base position: {self._robot_base_pos}")
        
        # 转换四元数格式：从 [w, x, y, z] 到 [x, y, z, w]
        base_quat_sapien = [base_quat[1], base_quat[2], base_quat[3], base_quat[0]]

        # 设置机器人基座位置
        if self.robot is not None:
            base_pose = sapien.Pose(robot_base_pos, base_quat_sapien)
            self.robot.left_entity.set_root_pose(base_pose)
            self.robot.right_entity.set_root_pose(base_pose)
            
            self.robot.left_entity_origion_pose = base_pose
            self.robot.right_entity_origion_pose = base_pose
            
            print(f"Robot base position set to: {robot_base_pos}")

        # 设置相机位置（应用世界Z偏移并转换坐标系）
        cam_pos_shifted = cam_pos.copy()
        # cam_pos_shifted[0] += 0.25
        cam_pos_shifted[2] += self.world_z_offset
        
        # 使用转换函数设置相机姿态
        camera_pose = self._opencv_to_sapien_pose(cam_pos_shifted, rotation_matrix)
        print(f"Setting camera pose: Position {cam_pos_shifted}, Pose {camera_pose}")
        self.camera.set_pose(camera_pose)

        self.robot.set_planner(self.scene)
        print(f"Robot base set to: Position {robot_base_pos}, Yaw {np.rad2deg(yaw):.2f}°")

    def adjust_position_along_z_axis(self, position: np.ndarray, rotation: np.ndarray, distance: float = -0.09):
        """
        沿着手腕的z轴方向调整位置（默认缩小8cm）
        
        Args:
            position: 位置坐标
            rotation: 旋转矩阵或四元数
            distance: 调整距离，负值表示缩小距离（默认-0.08，即缩小8cm）
            
        Returns:
            np.ndarray: 调整后的位置
        """
        # 处理旋转
        if rotation.size == 9:  # 旋转矩阵
            rot_matrix = rotation.reshape(3, 3)
        elif rotation.size == 4:  # 四元数 [w, x, y, z]
            rot_matrix = t3d_quat.quat2mat(rotation)
        else:
            print(f"Invalid rotation format: {rotation.shape}")
            return position
            
        # 获取z轴方向（旋转矩阵的第三列）
        z_axis = rot_matrix[:, 2]
        
        # 沿z轴方向调整位置
        adjusted_position = position.copy()
        adjusted_position[:3] += distance * z_axis
        
        print(f"原始位置: {position[:3]}")
        print(f"Z轴方向: {z_axis}")
        print(f"调整后位置: {adjusted_position[:3]}")
        
        return adjusted_position
    
    def set_arm_poses(self, 
                     left_end_pos: Union[list, np.ndarray],
                     left_end_rot: Union[list, np.ndarray],
                     right_end_pos: Union[list, np.ndarray], 
                     right_end_rot: Union[list, np.ndarray]):
        """设置双臂末端位置和旋转"""
        if self.robot is None:
            print("Robot not loaded, cannot set arm poses")
            return
        
        try:
            # 处理左臂目标姿态
            left_pos = np.array(left_end_pos).flatten()
            right_pos = np.array(right_end_pos).flatten()
            
            # 处理旋转
            left_end_rot_np = np.array(left_end_rot)
            right_end_rot_np = np.array(right_end_rot)
            
            # 沿着手腕z轴方向缩小8cm
            left_pos = self.adjust_position_along_z_axis(left_pos, left_end_rot_np)
            right_pos = self.adjust_position_along_z_axis(right_pos, right_end_rot_np)
            
            # 获取机器人基座位置（如果已经设置过）
            if hasattr(self, '_robot_base_pos'):
                robot_base_pos = self._robot_base_pos
                print(f"Adding robot base offset: {robot_base_pos}")
                
                # 将基座坐标加到左右手坐标上
                left_pos[:3] += robot_base_pos
                right_pos[:3] += robot_base_pos
                
                print(f"左手腕+基座位置: {left_pos[:3]}")
                print(f"右手腕+基座位置: {right_pos[:3]}")
            else:
                print("Warning: Robot base position not set, using original coordinates")
                            
            # 处理左臂旋转
            left_end_rot = np.array(left_end_rot)
            if left_end_rot.size == 9:  # 旋转矩阵
                left_rot_matrix = left_end_rot.reshape(3, 3)
                left_quat = t3d_quat.mat2quat(left_rot_matrix)
            elif left_end_rot.size == 4:  # 四元数
                left_quat = left_end_rot.flatten()
            else:
                print(f"Invalid left rotation format: {left_end_rot.shape}")
                left_quat = np.array([1, 0, 0, 0])
            
            # 处理右臂目标姿态
            
            # 处理右臂旋转
            right_end_rot = np.array(right_end_rot)
            if right_end_rot.size == 9:  # 旋转矩阵
                right_rot_matrix = right_end_rot.reshape(3, 3)
                right_quat = t3d_quat.mat2quat(right_rot_matrix)
            elif right_end_rot.size == 4:  # 四元数
                right_quat = right_end_rot.flatten()
            else:
                print(f"Invalid right rotation format: {right_end_rot.shape}")
                right_quat = np.array([1, 0, 0, 0])
            
            # 确保四元数格式正确 [w, x, y, z]
            if left_quat.size != 4:
                left_quat = np.array([1, 0, 0, 0])
            if right_quat.size != 4:
                right_quat = np.array([1, 0, 0, 0])
            
            # 构造目标姿态 [x, y, z, qx, qy, qz, qw]
            left_target_pose = np.concatenate([left_pos[:3], left_quat[1:4], left_quat[0:1]])
            right_target_pose = np.concatenate([right_pos[:3], right_quat[1:4], right_quat[0:1]])
            
            print(f"Planning path for left arm to target: {left_target_pose}")
            print(f"Planning path for right arm to target: {right_target_pose}")
            
            # 使用机器人的路径规划功能
            print("########### LEFT: ##############")
            left_plan_result = self.robot.left_plan_path(
                target_pose=left_target_pose.tolist(),
                constraint_pose=None,
                use_point_cloud=False,
                use_attach=False
            )
            print("########### RIGHT: ##############")
            right_plan_result = self.robot.right_plan_path(
                target_pose=right_target_pose.tolist(),
                constraint_pose=None,
                use_point_cloud=False,
                use_attach=False
            )
            
            print(f"Left planning result: {left_plan_result.get('status', 'Unknown')}")
            print(f"Right planning result: {right_plan_result.get('status', 'Unknown')}")
            
            left_path = left_plan_result.get('position', [])
            right_path = right_plan_result.get('position', [])
                
            if len(left_path) > 0:
                left_target_joints = left_path[-1]
                print(f"Executing joint positions - Left: {left_target_joints}")
                self.robot.set_arm_joints(
                    target_position=left_target_joints,
                    target_velocity=[0.0] * len(left_target_joints),
                    arm_tag="left"
                )

            if len(right_path) > 0:
                right_target_joints = right_path[-1]
                print(f"Executing joint positions - Right: {right_target_joints}")                
                self.robot.set_arm_joints(
                    target_position=right_target_joints,
                    target_velocity=[0.0] * len(right_target_joints),
                    arm_tag="right"
                )
                
            # 运行仿真
            for _ in range(10000):
                self.scene.step()
                
            # 验证最终位置
            final_left_pose = self.robot.get_left_endpose()
            final_right_pose = self.robot.get_right_endpose()
                
            print(f"Final left end pose: {final_left_pose}")
            print(f"Final right end pose: {final_right_pose}")
                
        except Exception as e:
            print(f"Error in set_arm_poses: {e}")
            import traceback
            traceback.print_exc()
            try:
                self._set_simplified_arm_poses(np.array(left_end_pos), np.array([1, 0, 0, 0]), 
                                              np.array(right_end_pos), np.array([1, 0, 0, 0]))
            except Exception:
                print("Simplified arm poses also failed")

    def set_arm_poses_from_world_frame(self,
                                        left_pos_world: Union[list, np.ndarray],
                                        left_rot_world: Union[list, np.ndarray],
                                        right_pos_world: Union[list, np.ndarray],
                                        right_rot_world: Union[list, np.ndarray]) -> None:
        """直接使用世界坐标系下的手腕位姿设置机械臂目标位姿"""
        if self.robot is None:
            print("Robot not loaded, cannot set arm poses")
            return

        try:
            left_pos_world = np.array(left_pos_world, dtype=float).reshape(3)
            right_pos_world = np.array(right_pos_world, dtype=float).reshape(3)

            self.set_arm_poses(
                left_end_pos=left_pos_world.tolist(),
                left_end_rot=np.array(left_rot_world, dtype=float),
                right_end_pos=right_pos_world.tolist(),
                right_end_rot=np.array(right_rot_world, dtype=float),
            )
        except Exception as e:
            print(f"Error in set_arm_poses_from_world_frame: {e}")
    
    def _set_simplified_arm_poses(self, left_pos, left_quat, right_pos, right_quat):
        print("Setting simplified arm poses (placeholder implementation)")
        pass
    
    def render_frame(self) -> np.ndarray:
        """
        渲染当前帧
        
        Returns:
            np.ndarray: RGB图像 (H, W, 3)，数据类型为uint8
        """
        # 更新场景
        self.scene.step()
        self.scene.update_render()
        
        # 获取相机图像
        self.camera.take_picture()
        camera_rgba = self.camera.get_picture("Color")
        camera_rgba_img = (camera_rgba * 255).clip(0, 255).astype("uint8")
        
        # 获取RGB图像（去掉alpha通道）
        rgb = camera_rgba_img[:, :, :3]
        rgb = cv2.flip(rgb, -1)
        
        return rgb

    def render_and_save(self, head_path: str):
        """渲染并将头部相机图像保存到文件"""
        # 更新场景与渲染
        self.scene.step()
        self.scene.update_render()

        # 头部相机
        self.camera.take_picture()
        head_rgba = self.camera.get_picture("Color")
        head_img = (head_rgba * 255).clip(0, 255).astype("uint8")[:, :, :3]
        # 上下翻转（flipCode=0 表示垂直翻转） 1 水平， -1 水平+垂直
        head_img = cv2.flip(head_img, -1)
                  
        cv2.imwrite(head_path, cv2.cvtColor(head_img, cv2.COLOR_RGB2BGR))
        print(f"Head camera image saved to: {head_path}")
    
    def render_scene(self, 
                    camera_extrinsics: Dict[str, Any],
                    left_end_pos: Union[list, np.ndarray],
                    left_end_rot: Union[list, np.ndarray],
                    right_end_pos: Union[list, np.ndarray],
                    right_end_rot: Union[list, np.ndarray]) -> np.ndarray:
        """一键渲染：设置所有参数并渲染场景"""
        # 设置机器人基座位置
        self.set_robot_base_pose(camera_extrinsics)
        
        # 初始化关节状态
        self._init_joint_states()
        
        # 设置双臂位置
        self.set_arm_poses(left_end_pos, left_end_rot, right_end_pos, right_end_rot)
        
        # 渲染并返回图像
        return self.render_frame()
    
    def show_viewer(self):
        """显示可视化窗口（如果启用了viewer）"""
        if self.enable_viewer and hasattr(self, 'viewer'):
            self.viewer.render()
    
    def save_image(self, image: np.ndarray, filepath: str):
        """保存图像到文件"""
        cv2.imwrite(filepath, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        print(f"Image saved to: {filepath}")
    
    def close(self):
        """清理资源"""
        if hasattr(self, 'robot') and self.robot is not None:
            # 清理机器人相关的进程和连接
            if hasattr(self.robot, 'communication_flag') and self.robot.communication_flag:
                if hasattr(self.robot, 'left_conn') and self.robot.left_conn:
                    self.robot.left_conn.close()
                if hasattr(self.robot, 'right_conn') and self.robot.right_conn:
                    self.robot.right_conn.close()
                if hasattr(self.robot, 'left_proc') and self.robot.left_proc.is_alive():
                    self.robot.left_proc.terminate()
                if hasattr(self.robot, 'right_proc') and self.robot.right_proc.is_alive():
                    self.robot.right_proc.terminate()
        
        if hasattr(self, 'scene'):
            # 清理场景中的所有actor
            for actor in self.scene.get_all_actors():
                self.scene.remove_actor(actor)
        
        # 清理SAPIEN缓存
        sapien_clear_cache()
        
        print("Robot Renderer closed.")
    
    def __del__(self):
        """析构函数"""
        self.close()


def calculate_gripper_pose(thumb_tip_pos, index_tip_pos, index_joint_pos):
    """
    计算夹爪的位置和朝向
    
    Args:
        thumb_tip_pos: 大拇指尖位置
        index_tip_pos: 食指尖位置
        index_joint_pos: 食指关节位置
    
    Returns:
        tuple: (gripper_position, gripper_rotation_matrix)
            - gripper_position: 夹爪位置（拇指尖和食指尖的中点）
            - gripper_rotation_matrix: 夹爪旋转矩阵，列向量分别为x, y, z轴
                - z轴: 指向夹爪闭合/张开方向（食指尖到拇指尖）
                - y轴: 由拇指尖、食指尖、食指关节三点定义的
                  平面的法向量
                - x轴: 通过右手规则由y和z轴叉乘得到
    """
    # 确保输入为numpy数组
    thumb_tip_pos = np.array(thumb_tip_pos, dtype=float)
    index_tip_pos = np.array(index_tip_pos, dtype=float)
    index_joint_pos = np.array(index_joint_pos, dtype=float)
    
    # 1. 计算夹爪位置（拇指尖和食指尖的中点）
    gripper_position = 0.5 * (thumb_tip_pos + index_tip_pos)
    
    # 2. 计算z轴：指尖-指尖的方向（食指尖到拇指尖，夹爪闭合方向）
    z_axis = thumb_tip_pos - index_tip_pos
    z_axis = z_axis / (np.linalg.norm(z_axis) + 1e-9)  # 归一化
    
    # 3. 计算y轴：由拇指尖、食指尖、食指关节三点定义的平面的法向量
    # 计算两个向量：v1 = 食指尖 -> 拇指尖，v2 = 食指尖 -> 食指关节
    v1 = thumb_tip_pos - index_tip_pos
    v2 = index_joint_pos - index_tip_pos
    
    # y轴是v1和v2的叉乘，表示平面法向量
    y_axis = np.cross(v1, v2)
    y_axis = y_axis / (np.linalg.norm(y_axis) + 1e-9)  # 归一化
    
    # 4. 计算x轴：通过右手规则，保证坐标系正交性
    x_axis = np.cross(y_axis, z_axis)
    x_axis = x_axis / (np.linalg.norm(x_axis) + 1e-9)  # 归一化
    
    # 5. 构建旋转矩阵，列向量分别为x, y, z轴
    rotation_matrix = np.column_stack([x_axis, y_axis, z_axis])
    
    return gripper_position, rotation_matrix


def generate_robot_video(json_path: str, 
                        output_video_path: str = "robot_animation.mp4",
                        num_frames: int = 10,
                        fps: int = 10,
                        task_name: str = "clean_cups"):
    """
    生成机器人动作视频
    
    Args:
        json_path: JSON数据文件路径
        output_video_path: 输出视频文件路径
        num_frames: 要渲染的帧数
        fps: 视频帧率
    """
    
    # 加载JSON数据
    with open(json_path, "r") as f:
        data = json.load(f)
        
    # 初始化结果数据结构
    default_intrinsic = np.array([
        [736.6339111328125, 0.0, 960.0],
        [0.0, 736.6339111328125, 540.0],
        [0.0, 0.0, 1.0]
    ], dtype=float).tolist()
    
    # 这里定义了一个结果数据结构示例，实际使用时需要填充并返回
    # 当前版本中我们直接使用data中的数据，不需要重新构建result

    # 检查可用帧数
    available_frames = len(data["camera"]["transforms"])
    actual_frames = min(num_frames, available_frames)
    print(f"总共可用帧数: {available_frames}, 将渲染前 {actual_frames} 帧")

    # 使用固定相机内参矩阵
    camera_intrinsics = np.array([
        [736.6339111328125, 0.0, 960.0],
        [0.0, 736.6339111328125, 540.0],
        [0.0, 0.0, 1.0]
    ], dtype=float)
    print(f"使用固定相机内参矩阵:\n{camera_intrinsics}")

    # 从内参矩阵提取焦距和主点
    fy = camera_intrinsics[1, 1]
    cx = camera_intrinsics[0, 2]
    cy = camera_intrinsics[1, 2]
    
    # 计算视场角（垂直方向）
    image_height = data["camera"].get("height", 1080)
    fovy_rad = 2 * np.arctan(image_height / (2 * fy))
    fovy_deg = np.rad2deg(fovy_rad)
    print(f"从固定内参计算的视场角: {fovy_deg:.2f}°")
    
    # 获取图像尺寸
    image_width = int(2*cx)
    image_height = int(2*cy)
    
    # 初始化渲染器
    renderer = RobotRenderer(
        image_width=image_width,
        image_height=image_height,
        enable_viewer=False,  # 视频生成时不启用viewer
        fovy_deg=fovy_deg,
        arms_z_offset=0.9,
    )

    # 初始化视频编写器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (image_width, image_height))
    
    # 注：仅在左上角叠加帧号信息
    
    try:
        # 处理每一帧
        for frame_idx in range(actual_frames):
            print(f"\n=== 处理第 {frame_idx + 1}/{actual_frames} 帧 ===")
            
            # 读取当前帧的相机外参
            T_cv = np.array(data["camera"]["transforms"][frame_idx], dtype=float)
            camera_position = T_cv[:3, 3]
            camera_rotation = T_cv[:3, :3]
            
            # 坐标系转换
            opencv_to_sapien = np.array([
                [-1,  0,  0],
                [0,  -1,  0],
                [0,  0,  1]
            ])        
            camera_rotation = camera_rotation @ opencv_to_sapien
            
            # 调整相机位置，提高高度
            camera_position[2] += 1.6
            camera_extrinsics = {"position": camera_position.tolist(), "rotation": camera_rotation.tolist()}


            
            # 读取左右手的拇指尖、食指尖和食指关节位置
            left_thumb_tip = np.array(
                data["left"]["thumbTip"]["position"][frame_idx], dtype=float)
            left_index_tip = np.array(
                data["left"]["indexFingerTip"]["position"][frame_idx], dtype=float)
            left_index_joint = np.array(
                data["left"]["indexFingerIntermediateTip"]["position"][frame_idx], 
                dtype=float)
            
            right_thumb_tip = np.array(
                data["right"]["thumbTip"]["position"][frame_idx], dtype=float)
            right_index_tip = np.array(
                data["right"]["indexFingerTip"]["position"][frame_idx], dtype=float)
            right_index_joint = np.array(
                data["right"]["indexFingerIntermediateTip"]["position"][frame_idx], 
                dtype=float)
            
            # 计算夹爪位置和旋转矩阵
            left_pos_world, left_rot_world = calculate_gripper_pose(
                left_thumb_tip, left_index_tip, left_index_joint)
            right_pos_world, right_rot_world = calculate_gripper_pose(
                right_thumb_tip, right_index_tip, right_index_joint)
            
            print(f"计算的左手夹爪位置: {left_pos_world}")
            print(f"计算的右手夹爪位置: {right_pos_world}")
            
            # 坐标系变换
            xy_to_yx = np.array([
                [0,  -1,  0],
                [-1,  0,  0],
                [0,  0,  1]
            ])     
            # 应用坐标系变换
            left_pos_world = left_pos_world @ xy_to_yx
            right_pos_world = right_pos_world @ xy_to_yx
            
            # # 同样需要变换旋转矩阵  ####################??????????????????????????#####################################
            # left_rot_world = left_rot_world @ xy_to_yx
            # right_rot_world = right_rot_world @ xy_to_yx
            
            # 应用缩放因子
            left_pos_world[:3] *= 1.1
            right_pos_world[:3] *= 1.1
            print(f"倍数后 世界系左腕位置: {left_pos_world}")
            print(f"倍数后世界系右腕位置: {right_pos_world}")        
            left_pos_world[0] += 1.6 #1.55#1.6 #1.3 # 本来就是[-0.8,-1.2]左右
            right_pos_world[0] += 1.6 # 1.55 #1.6 #1.3 
            left_pos_world[1] -= 0.15 #+=  #0.5 # (clean surface 0.5) # 0.8  (clean cupd的深度0.8->1.14) # +=0.15
            right_pos_world[1] -=0.15 #+=  # 0.5 # 0.8(clean cupd 0.8 感觉才是最符合实际的) # -=0.15
            left_pos_world[2] += 1.55  # 1.5
            right_pos_world[2] += 1.55 #1.5  # 1.2     
                     
            print(f"Frame {frame_idx}: 相机位置: {camera_position}")
            print(f"Frame {frame_idx}: 左腕位置: {left_pos_world}")
            print(f"Frame {frame_idx}: 右腕位置: {right_pos_world}")
            

            # R_y180_left = np.array([[0, 1, 0],
            #                         [ 0, 0, 1],
            #                         [ -1, 0, 0]], dtype=float)
            # R_y180_right = np.array([[0, -1, 0],
            #                         [ 0, 0, -1],
            #                         [ 1, 0, 0]], dtype=float)
            # R_y180_left = np.array([[0, -1, 0],
            #                         [ 0, 0, 1],
            #                         [ -1, 0, 0]], dtype=float)
            # R_y180_right = np.array([[0, 1, 0],
            #                         [0, 0, 1],
            #                         [1, 0, 0]], dtype=float)

            # # 应用旋转调整以匹配机器人夹爪方向
            # R_y180_left = np.array([[0, -1, 0],
            #                         [0, 0, -1],
            #                         [-1, 0, 0]], dtype=float)
            # R_y180_right = np.array([[0, 1, 0],
            #                         [0, 0, -1],
            #                         [1, 0, 0]], dtype=float)
            R_y180_left = np.array([[0, 0, 1],
                                    [ 1, 0, 0], # -z 10:10
                                    [ 0, -1, 0]], dtype=float)
            R_y180_right = np.array([[0, 0, 1],
                                    [1, 0, 0],
                                    [0, -1, 0]], dtype=float)
    
            left_rot_world = R_y180_left @ left_rot_world
            right_rot_world = R_y180_right @ right_rot_world

            # 只在第一帧设置机器人基座位置
            if frame_idx == 0:
                renderer.set_robot_base_pose(camera_extrinsics)

            # 设置当前帧的手臂位姿
            renderer.set_arm_poses_from_world_frame(
                left_pos_world=left_pos_world,
                left_rot_world=left_rot_world,
                right_pos_world=right_pos_world,
                right_rot_world=right_rot_world,
            )

            # 渲染当前帧
            rgb_image = renderer.render_frame()
            
            # 转换为BGR格式
            bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)

            # 左上角叠加帧号信息
            frame_text = f"Frame {frame_idx + 1}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1
            thickness = 1
            x0, y0 = 10, 10
            (text_size, baseline) = cv2.getTextSize(frame_text, font, font_scale, thickness)
            # 不再绘制黑色背景，文本颜色改为黑色
            cv2.putText(bgr_image, frame_text, (x0, y0 + text_size[1] + 2), font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)
            video_writer.write(bgr_image)
            
            # 可选：保存每一帧为图片（用于调试）
            frame_dir = f"code_painting/{task_name}/B_case13_1"
            if not os.path.exists(frame_dir):
                os.makedirs(frame_dir)
            frame_path = f"{frame_dir}/frame_{frame_idx:04d}.png"
            cv2.imwrite(frame_path, bgr_image)
            print(f"Frame {frame_idx} saved to {frame_path}")
        
        print(f"\n视频生成完成！保存到: {output_video_path}")
        print(f"视频参数: {actual_frames}帧, {fps}fps, 分辨率{image_width}x{image_height}")
        
    except Exception as e:
        print(f"视频生成过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        # 清理资源
        video_writer.release()
        renderer.close()


def demo_usage(frame_idx=0):
    """演示用法 - 单帧渲染"""
    import numpy as np
    import json

    # json_path = "/home/pine/RoboTwin2/code_painting/clean_surface/0_wrist_data.json"
    json_path = "/home/pine/RoboTwin2/code_painting/clean_cups/0_wrist_data.json"
    # json_path = "/home/pine/RoboTwin2/code_painting/assemble_disassemble_furniture_bench_lamp/0_wrist_data.json"
    
    with open(json_path, "r") as f:
        data = json.load(f)

    # 使用固定相机内参矩阵
    camera_intrinsics = np.array([
        [736.6339111328125, 0.0, 960.0],
        [0.0, 736.6339111328125, 540.0],
        [0.0, 0.0, 1.0]
    ], dtype=float)
    print(f"使用固定相机内参矩阵:\n{camera_intrinsics}")

    # 从内参矩阵提取焦距和主点
    fy = camera_intrinsics[1, 1]
    cx = camera_intrinsics[0, 2]
    cy = camera_intrinsics[1, 2]
    
    # 计算视场角（垂直方向）
    image_height = data["camera"].get("height", 1080)
    fovy_rad = 2 * np.arctan(image_height / (2 * fy))
    fovy_deg = np.rad2deg(fovy_rad)
    print(f"从固定内参计算的视场角: {fovy_deg:.2f}°")

    # 读取指定帧相机外参
    T_cv = np.array(data["camera"]["transforms"][frame_idx], dtype=float)
    print(f"原始相机外参矩阵 T_cv (第{frame_idx}帧):\n{T_cv}")

    # 提取相机位置和旋转
    camera_position = T_cv[:3, 3]
    camera_rotation = T_cv[:3, :3]
    
    # 绕xy轴分别旋转180度
    opencv_to_sapien = np.array([
        [-1,  0,  0],  # X轴保持不变 c 
        [0,  -1,  0],  # Y轴翻转：下 -> 上 (简单朝左偏了)
        [0,  0,  1]   # Z轴翻转：前 -> 后
    ])        

    # 转换旋转矩阵
    camera_rotation = camera_rotation @ opencv_to_sapien    
    
    
    print(f"OpenCV相机外参矩阵位置:\n{camera_position}")
    print(f"OpenCV相机外参矩阵旋转:\n{camera_rotation}")
    
    # 调整相机位置，提高高度
    camera_position[2] += 1.3
    
    camera_extrinsics = {"position": camera_position.tolist(), "rotation": camera_rotation.tolist()}

    # 读取左右手指定帧末端位姿
    body_name = ["wrists", "thumbTip", "indexFingerTip"]
    body_name_only_pos = ["thumbIntermediateTip", "indexFingerIntermediateTip", "thumbIntermediateBase", "indexFingerIntermediateBase"]    
                        
    left_wrists_pos_world = np.array(data["left"]["wrists"]["position"][frame_idx], dtype=float)
    left_wrists_rot_world = np.array(data["left"]["wrists"]["orientation"][frame_idx], dtype=float)
    left_thumbTip_pos_world = np.array(data["left"]["thumbTip"]["position"][frame_idx], dtype=float)
    left_thumbTip_rot_world = np.array(data["left"]["thumbTip"]["orientation"][frame_idx], dtype=float)
    left_indexFingerTip_pos_world = np.array(data["left"]["indexFingerTip"]["position"][frame_idx], dtype=float)
    left_indexFingerTip_rot_world = np.array(data["left"]["indexFingerTip"]["orientation"][frame_idx], dtype=float)
    
    right_wrists_pos_world = np.array(data["right"]["wrists"]["position"][frame_idx], dtype=float)
    right_wrists_rot_world = np.array(data["right"]["wrists"]["orientation"][frame_idx], dtype=float)
    right_thumbTip_pos_world = np.array(data["right"]["thumbTip"]["position"][frame_idx], dtype=float)
    right_thumbTip_rot_world = np.array(data["right"]["thumbTip"]["orientation"][frame_idx], dtype=float)
    right_indexFingerTip_pos_world = np.array(data["right"]["indexFingerTip"]["position"][frame_idx], dtype=float)
    right_indexFingerTip_rot_world = np.array(data["right"]["indexFingerTip"]["orientation"][frame_idx], dtype=float)
    
    left_indexFingerIntermediateTip_pos_world = np.array(data["left"]["indexFingerIntermediateTip"]["position"][frame_idx], dtype=float)
    right_indexFingerIntermediateTip_pos_world = np.array(data["right"]["indexFingerIntermediateTip"]["position"][frame_idx], dtype=float)
    # 分析手腕朝向和欧拉角
    print("\n=== 手腕朝向分析 ===")
    print(f"世界系左腕初始位置: {left_wrists_pos_world}")
    print(f"世界系右腕初始位置: {right_wrists_pos_world}")
    print(f"世界系左腕初始四元数: {left_wrists_rot_world}")
    print(f"世界系右腕初始四元数: {right_wrists_rot_world}")
    
    # 计算夹爪位置和旋转矩阵
    left_pos_world, left_rot_world = calculate_gripper_pose(
        left_thumbTip_pos_world, 
        left_indexFingerTip_pos_world, 
        left_indexFingerIntermediateTip_pos_world)
    right_pos_world, right_rot_world = calculate_gripper_pose(
        right_thumbTip_pos_world, 
        right_indexFingerTip_pos_world, 
        right_indexFingerIntermediateTip_pos_world)
    
    print("\n=== 夹爪位姿分析 ===")
    print(f"左手夹爪位置: {left_pos_world}")
    print(f"右手夹爪位置: {right_pos_world}")
    print(f"左手夹爪旋转矩阵:\n{left_rot_world}")
    print(f"右手夹爪旋转矩阵:\n{right_rot_world}")
    
    # # 旋转矩阵到欧拉角转换
    # try:
    #     # 检查输入格式并转换
    #     if left_rot_world.shape == (3, 3):
    #         # 如果是旋转矩阵，先转换为四元数
    #         left_quat = t3d_quat.mat2quat(left_rot_world)
    #         right_quat = t3d_quat.mat2quat(right_rot_world)
    #     else:
    #         # 如果已经是四元数，直接使用
    #         left_quat = left_rot_world
    #         right_quat = right_rot_world
            
    #     # 四元数转欧拉角 (ZYX顺序)
    #     left_euler = t3d_euler.quat2euler(left_quat, 'rzyx')
    #     right_euler = t3d_euler.quat2euler(right_quat, 'rzyx')
        
    #     print(f"\n左腕欧拉角 (度): roll={np.rad2deg(left_euler[0]):.1f}°, pitch={np.rad2deg(left_euler[1]):.1f}°, yaw={np.rad2deg(left_euler[2]):.1f}°")
    #     print(f"右腕欧拉角 (度): roll={np.rad2deg(right_euler[0]):.1f}°, pitch={np.rad2deg(right_euler[1]):.1f}°, yaw={np.rad2deg(right_euler[2]):.1f}°")
        
    #     # 朝向分析
    #     print("\n左腕朝向:")
    #     if abs(left_euler[0]) < 0.1:
    #         print("  Roll: 基本水平")
    #     elif left_euler[0] > 0:
    #         print("  Roll: 向右倾斜")
    #     else:
    #         print("  Roll: 向左倾斜")
        
    #     if abs(left_euler[1]) < 0.1:
    #         print("  Pitch: 基本水平")
    #     elif left_euler[1] > 0:
    #         print("  Pitch: 向上倾斜")
    #     else:
    #         print("  Pitch: 向下倾斜")
        
    #     if abs(left_euler[2]) < 0.1:
    #         print("  Yaw: 朝前")
    #     elif left_euler[2] > 0:
    #         print("  Yaw: 向右偏转")
    #     else:
    #         print("  Yaw: 向左偏转")
        
    #     print("\n右腕朝向:")
    #     if abs(right_euler[0]) < 0.1:
    #         print("  Roll: 基本水平")
    #     elif right_euler[0] > 0:
    #         print("  Roll: 向右倾斜")
    #     else:
    #         print("  Roll: 向左倾斜")
        
    #     if abs(right_euler[1]) < 0.1:
    #         print("  Pitch: 基本水平")
    #     elif right_euler[1] > 0:
    #         print("  Pitch: 向上倾斜")
    #     else:
    #         print("  Pitch: 向下倾斜")
        
    #     if abs(right_euler[2]) < 0.1:
    #         print("  Yaw: 朝前")
    #     elif right_euler[2] > 0:
    #         print("  Yaw: 向右偏转")
    #     else:
    #         print("  Yaw: 向左偏转")
    #     print("==================\n")
    # except Exception as e:
    #     print(f"Error in wrist orientation analysis: {e}")
    
    # 读取当前帧的左右手末端位姿
    # 使用三点（拇指尖T、食指尖I、手腕W）计算抓取姿态
    # 左手
    # left_thumb_tip = np.array(data["left"]["thumbTip"]["position"][frame_idx], dtype=float)
    # left_index_tip = np.array(data["left"]["indexFingerTip"]["position"][frame_idx], dtype=float)
    # left_wrist = np.array(data["left"]["wrists"]["position"][frame_idx], dtype=float)
    
    # 使用新函数计算夹爪位姿
    print("\n=== 使用新函数计算夹爪位姿 ===")
    print(f"世界系左腕初始位置: {left_pos_world}")
    print(f"世界系右腕初始位置: {right_pos_world}")
    xy_to_yx = np.array([
        [0,  -1,  0],
        [-1,  0,  0],
        [0,  0,  1]
    ])     
    left_pos_world = left_pos_world @ xy_to_yx
    right_pos_world = right_pos_world @ xy_to_yx
    print(f"转换后世界系左腕位置: {left_pos_world}")
    print(f"转换后世界系右腕位置: {right_pos_world}")    
    # left_pos_world[0] += 1.55
    # right_pos_world[0] += 1.55    
    # left_pos_world[1] += 0.95
    # right_pos_world[1] += 0.9              
    # left_pos_world[2] += 1.6
    # right_pos_world[2] += 1.6
    
    # clean cups world_base_pose:  [-0.03613232  1.220203    0.]
    # 假设比人手长1.2倍
    left_pos_world[:3] *= 1.1
    right_pos_world[:3] *= 1.1
    print(f"倍数后 世界系左腕位置: {left_pos_world}")
    print(f"倍数后世界系右腕位置: {right_pos_world}")        
    left_pos_world[0] += 1.7 #1.55#1.6 #1.3 # 本来就是[-0.8,-1.2]左右
    right_pos_world[0] += 1.7 # 1.55 #1.6 #1.3 
    left_pos_world[1] -= 0.2 #+=  #0.5 # (clean surface 0.5) # 0.8  (clean cupd的深度0.8->1.14) # +=0.15
    right_pos_world[1] -=0.2 #+=  # 0.5 # 0.8(clean cupd 0.8 感觉才是最符合实际的) # -=0.15
    left_pos_world[2] += 1.55  # 1.5
    right_pos_world[2] += 1.55 #1.5  # 1.2    
           
    
    print(f"相机位置: {camera_position}")
    print(f"世界系左腕位置: {left_pos_world}")
    print(f"世界系右腕位置: {right_pos_world}")
    
    print("===============左右手朝向绕Y轴变换180度===============")
    # 原来的在手腕上
    # R_y180_left = np.array([[0, -1, 0],
    #                         [ 0, 0, 1],
    #                         [ -1, 0, 0]], dtype=float)
    # R_y180_right = np.array([[0, 1, 0],
    #                          [0, 0, 1],
    #                          [1, 0, 0]], dtype=float)


    R_y180_left = np.array([[0, 0, -1],
                            [ 1, 0, 0], # -z 10:10
                            [ 0, -1, 0]], dtype=float)
    R_y180_right = np.array([[0, 0, -1],
                            [1, 0, 0],
                            [0, -1, 0]], dtype=float)
    # 左右手位置变换
    left_rot_world  = R_y180_left @ left_rot_world
    right_rot_world = R_y180_right @ right_rot_world

    # # 设置手臂朝前的旋转矩阵
    # # 朝前意味着末端执行器的Z轴指向正前方（+Y方向）
    # forward_rotation_euler = [np.pi, 0, 0]  # 绕Z轴旋转90度
    # # forward_rotation_euler = [0, np.pi, 0]  # 绕Z轴旋转90度
    # forward_rot_mat = t3d_euler.euler2mat(*forward_rotation_euler, 'sxyz')
    # left_rot_world = forward_rot_mat
    # right_rot_world = forward_rot_mat
    # print(f"世界系左腕向前旋转矩阵: \n{left_rot_world}")
    # print(f"世界系右腕向前旋转矩阵: \n{right_rot_world}")
    
    # 获取图像尺寸
    image_width = int(2*cx)
    image_height = int(2*cy)
    
    # 初始化渲染器
    renderer = RobotRenderer(
        image_width=image_width,
        image_height=image_height,
        enable_viewer=True,
        fovy_deg=fovy_deg,
        arms_z_offset=0.9,
    )

    try:
        # 设置机器人基座与相机
        renderer.set_robot_base_pose(camera_extrinsics)

        # 使用世界坐标的手腕位姿
        renderer.set_arm_poses_from_world_frame(
            left_pos_world=left_pos_world,
            left_rot_world=left_rot_world,
            right_pos_world=right_pos_world,
            right_rot_world=right_rot_world,
        )

        # 渲染图像
        rgb_image = renderer.render_frame()

        # 保存图像
        renderer.save_image(rgb_image, f"code_painting/robot_render_result_frame_{frame_idx}.png")
        renderer.render_and_save(head_path=f"code_painting/robot_render_result_frame_{frame_idx}_save.png")
        print(f"渲染结果已保存到 robot_render_result_frame_{frame_idx}.png")
        
        # 显示viewer
        if renderer.enable_viewer:
            print("Press Ctrl+C to exit viewer...")
            try:
                while True:
                    renderer.show_viewer()
            except KeyboardInterrupt:
                print("Exiting...")
        
    finally:
        renderer.close()
    
    print("Rendering complete.")


def demo_video_generation():
    """演示视频生成功能"""
    # JSON数据文件路径
    # task_name =["basic_fold"]
    num_frames = 10 #399 # 299
    task_name = "clean_cups" #"basic_pick_place" #"add_remove_lid"   # "assemble_disassemble_furniture_bench_lamp"  #"clean_surface" # "clean_cups"
    json_path = f"/home/pine/RoboTwin2/code_painting/{task_name}/0_wrist_data.json"
    
    # 输出视频路径
    output_video_path = f"code_painting/{task_name}/B_case13_1_robot_animation_{num_frames}frames.mp4"
    
    # 生成前10帧的视频
    generate_robot_video(
        json_path=json_path,
        output_video_path=output_video_path,
        num_frames=num_frames, #299, #10,
        fps=5,  # 5fps，播放较慢便于观察
        task_name=task_name
    )


if __name__ == "__main__":
    # 选择运行模式
    mode = input("选择运行模式 (1: 单帧渲染, 2: 视频生成): ").strip()
    
    if mode == "1":
        # 获取用户输入的帧索引
        frame_idx_input = input("请输入要渲染的帧索引 (0-based): ").strip()
        try:
            frame_idx = int(frame_idx_input)
            demo_usage(frame_idx=frame_idx)
        except ValueError:
            print("无效的帧索引，请输入一个整数。")
            demo_usage()
    elif mode == "2":
        demo_video_generation()
    else:
        print("无效选择，运行单帧渲染...")
        demo_usage()