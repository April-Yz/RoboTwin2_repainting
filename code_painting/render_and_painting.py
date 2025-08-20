#!/usr/bin/env python3
"""
机器人渲染器 - 用于cross painting任务的机器人姿态渲染
"""

import os
import numpy as np
import sapien.core as sapien
from sapien.render import clear_cache as sapien_clear_cache
from sapien.utils.viewer import Viewer
import transforms3d as t3d
from typing import Optional, Tuple, Union, Dict, Any
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
                 enable_third_person: bool = False,
                 third_person_distance: float = 1.5,  # 减小默认距离
                 third_person_height: float = 1.2,
                 third_person_yaw_offset_deg: float = -90.0,
                 third_person_pitch_deg: float = -30.0,  # 增大俯视角度
                 third_person_roll_deg: float = 0.0):  # 移除默认roll角度
        """
        初始化机器人渲染器
        
        Args:
            image_width: 渲染图像宽度
            image_height: 渲染图像高度
            enable_viewer: 是否启用可视化窗口
        """
        self.image_width = image_width
        self.image_height = image_height
        self.enable_viewer = enable_viewer
        # 视场角做合理范围限制，避免极端值导致渲染异常
        self.fovy_deg = float(np.clip(fovy_deg, 30.0, 120.0))
        # 第三视角相机配置
        self.enable_third_person = enable_third_person
        self.third_person_distance = third_person_distance
        self.third_person_height = third_person_height
        self.third_person_yaw_offset_deg = third_person_yaw_offset_deg
        self.third_person_pitch_deg = third_person_pitch_deg
        self.third_person_roll_deg = third_person_roll_deg
        self.side_camera = None
        
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
        
        # 添加地面
        self.scene.add_ground(0)
        
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
        # 这里需要根据您的具体机器人URDF文件路径进行调整
        # 假设您有aloha机器人的URDF文件
        try:
            # 尝试加载机器人URDF (需要根据实际路径调整)
            robot_urdf_path = "/data1/zjyang/program/third/RoboTwin/assets/embodiments/aloha-agilex/urdf/arx5_description_isaac.urdf"  # 请根据实际路径调整
            
            if os.path.exists(robot_urdf_path):
                # loader = self.scene.create_urdf_loader()
                # self.robot = loader.load(robot_urdf_path)
                with open("/data1/zjyang/program/third/RoboTwin/robot_config.json", "r") as f:
                    robot_cfg = json.load(f)
                    
                self.need_topp = True
                self.robot = Robot(self.scene, self.need_topp, **robot_cfg)
                # self.robot.set_planner(self.scene)
                # self.robot.set_root_pose(sapien.Pose([0, 0, 0]))
                self.robot.init_joints()
                print("Robot loaded successfully with dual-arm configuration")
            
                self.robot.print_info()
                
            else:
                # 如果没有URDF文件，创建简化的机器人模型
                print(f"!!! URDF file not found: {robot_urdf_path}. Creating simplified robot model...")
                # self._create_simplified_robot()
                
        except Exception as e:
            print(f"Failed to load robot URDF: {e}")
            print("Creating simplified robot model...")
            # self._create_simplified_robot()
        
        # 初始化关节状态
        self._init_joint_states()
    
    def _init_joint_states(self):
        """初始化关节状态"""
        if self.robot is not None:
            try:
                # 移动到家位置
                self.robot.move_to_homestate()
                
                # 设置初始夹爪状态 (打开)
                self.robot.left_gripper_val = 0.8  # 打开状态
                self.robot.right_gripper_val = 0.8  # 打开状态
                
                # 更新一步物理仿真以应用关节状态
                self.scene.step()
                # print("Robot joint states initialized to home position")
                # print(f"Left arm joints: {self.robot.get_left_arm_jointState()}")
                # print(f"Right arm joints: {self.robot.get_right_arm_jointState()}")
                
            except Exception as e:
                print(f"Failed to initialize joint states: {e}")
    
        
        # 设置简化模型的初始位置
        # self.robot.set_root_pose(sapien.Pose([0, 0, 0.1]))
        # self.left_arm.set_pose(sapien.Pose([0.1, 0.3, 0.5], [1, 0, 0, 0]))
        # self.right_arm.set_pose(sapien.Pose([0.1, -0.3, 0.5], [1, 0, 0, 0]))
        # self.left_gripper.set_pose(sapien.Pose([0.25, 0.3, 0.5], [1, 0, 0, 0]))
        # self.right_gripper.set_pose(sapien.Pose([0.25, -0.3, 0.5], [1, 0, 0, 0]))
        # self.head.set_pose(sapien.Pose([0, 0, 0.8], [1, 0, 0, 0]))
    
    def _setup_camera(self):
        """设置相机"""
        # 创建相机
        self.camera = self.scene.add_camera(
            name="head_camera",
            width=self.image_width,
            height=self.image_height,
            fovy=np.deg2rad(self.fovy_deg),
            near=0.01,
            far=100.0  # 增大远平面距离
        )
        
        # 设置默认相机位置 (头部相机) - 调整高度和角度以便更好地看到机器人
        # 使用四元数 [x, y, z, w] 表示旋转，这里设置为向下看的姿态
        self.camera.set_entity_pose(sapien.Pose([0, 0, 1.5], [0.7071, 0, 0.7071, 0]))

    def _compute_lookat_rotation(self, cam_pos: np.ndarray, target_pos: np.ndarray,
                                 up_hint: np.ndarray = np.array([0.0, 0.0, 1.0])) -> np.ndarray:
        """根据相机位置与目标点计算旋转矩阵，使相机前向为指向目标。
        约定OpenGL/Sapien相机前向为 -Z，因此令 R[:, 2] = forward_gl = -normalize(target - cam)。
        """
        forward = target_pos - cam_pos
        norm = np.linalg.norm(forward)
        if norm < 1e-8:
            forward_gl = np.array([0.0, 0.0, -1.0])
        else:
            forward_gl = -(forward / norm)
        right = np.cross(up_hint, forward_gl)
        r_norm = np.linalg.norm(right)
        if r_norm < 1e-8:
            # up 与 forward 共线时，调整 up
            up_hint = np.array([0.0, 1.0, 0.0])
            right = np.cross(up_hint, forward_gl)
            r_norm = np.linalg.norm(right)
        right = right / r_norm
        up = np.cross(forward_gl, right)
        R = np.stack([right, up, forward_gl], axis=1)
        return R

    def set_arm_poses_from_camera_frame(self,
                                        T_world_from_camera_cv: np.ndarray,
                                        left_pos_cam_cv: Union[list, np.ndarray],
                                        left_rot_cam_cv: Union[list, np.ndarray],
                                        right_pos_cam_cv: Union[list, np.ndarray],
                                        right_rot_cam_cv: Union[list, np.ndarray]) -> None:
        """
        直接从相机(OpenCV坐标系)下的手腕位姿设置机械臂目标位姿：
        世界手腕位姿 = 世界从相机(T_wc_cv) × 相机下手腕位姿(T_cwrist)

        Args:
            T_world_from_camera_cv: 4x4 外参矩阵 (OpenCV基底) world_from_camera
            left_pos_cam_cv: 左腕在相机(OpenCV)系下位置 [x,y,z]
            left_rot_cam_cv: 左腕在相机(OpenCV)系下朝向 (3x3 扁平或 3x3)
            right_pos_cam_cv: 右腕在相机(OpenCV)系下位置 [x,y,z]
            right_rot_cam_cv: 右腕在相机(OpenCV)系下朝向 (3x3 扁平或 3x3)
        """
        if self.robot is None:
            print("Robot not loaded, cannot set arm poses")
            return

        T_wc = np.array(T_world_from_camera_cv, dtype=float)
        if T_wc.shape != (4, 4):
            raise ValueError(f"T_world_from_camera_cv must be 4x4, got {T_wc.shape}")

        R_wc = T_wc[:3, :3]
        t_wc = T_wc[:3, 3]

        def to_rot_mat(rot_any: Union[list, np.ndarray]) -> np.ndarray:
            rot_any = np.array(rot_any, dtype=float)
            if rot_any.size == 9:
                return rot_any.reshape(3, 3)
            elif rot_any.shape == (3, 3):
                return rot_any
            else:
                print(f"Invalid rotation format {rot_any.shape}, fallback to identity")
                return np.eye(3)

        # 左腕世界位姿
        left_pos_cam_cv = np.array(left_pos_cam_cv, dtype=float).reshape(3)
        left_rot_cam_cv = to_rot_mat(left_rot_cam_cv)
        
        # # new for opengl
        # cv_to_gl = np.array([[1, 0, 0],
        #                     [0, -1, 0],
        #                     [0, 0, -1]], dtype=float)

        # left_pos_gl = cv_to_gl @ left_pos_cam_cv
        # left_rot_gl = cv_to_gl @ left_rot_cam_cv

        # left_pos_world = R_wc @ left_pos_gl + t_wc
        # left_rot_world = R_wc @ left_rot_gl

        left_pos_world = (R_wc @ left_pos_cam_cv + t_wc)
        left_rot_world = (R_wc @ left_rot_cam_cv)
        left_quat_wxyz = t3d_quat.mat2quat(left_rot_world)  # [w,x,y,z]

        # 右腕世界位姿
        right_pos_cam_cv = np.array(right_pos_cam_cv, dtype=float).reshape(3)
        right_rot_cam_cv = to_rot_mat(right_rot_cam_cv)
        right_pos_world = (R_wc @ right_pos_cam_cv + t_wc)
        right_rot_world = (R_wc @ right_rot_cam_cv)
        right_quat_wxyz = t3d_quat.mat2quat(right_rot_world)  # [w,x,y,z]

        # 交由既有接口
        self.set_arm_poses(
            left_end_pos=left_pos_world.tolist(),
            left_end_rot=left_quat_wxyz.tolist(),
            right_end_pos=right_pos_world.tolist(),
            right_end_rot=right_quat_wxyz.tolist(),
        )
    
    def set_robot_base_pose(self, camera_extrinsics: Dict[str, Any]):
        """
        根据相机外参设置机器人基座位置，智能修正相机外参确保机器人与地面平行
        
        Args:
            camera_extrinsics: 包含相机位置和旋转的字典
        """
        cam_pos = np.array(camera_extrinsics['position'])

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
        
        # 智能修正：只提取绕世界z轴的旋转分量
        # 方法1：通过旋转矩阵的前向投影提取yaw角
        # 相机的前向向量：在OpenGL约定下通常是 -Z 方向，这里取反以避免180°偏差
        camera_forward = rotation_matrix[:, 2]
        
        # 将前向向量投影到xy平面，计算yaw角
        forward_xy = camera_forward[:2]  # 只取x,y分量
        forward_xy_normalized = forward_xy / (np.linalg.norm(forward_xy) + 1e-8)  # 避免除零
        
        # 计算yaw角（相对于x轴的角度）
        yaw = np.arctan2(forward_xy_normalized[1], forward_xy_normalized[0])
        
        # 调整yaw角，使机器人在相机视野内
        yaw = yaw + np.pi  # 添加180度偏移，使机器人面向相机
        
        print(f"Camera forward vector: {camera_forward}")
        print(f"Extracted yaw angle: {np.rad2deg(yaw):.2f} degrees (after adjustment)")
        
        # 创建只包含yaw旋转的四元数（保持机器人直立）
        base_quat = t3d_euler.euler2quat(0, 0, yaw)  # roll=0, pitch=0, 只有yaw
        
        # 对于相机头部，我们保持原始的旋转
        head_quat = t3d_quat.mat2quat(rotation_matrix)
        
        # 计算机器人基座位置
        # 修改：减小相机到基座的偏移，使机器人更靠近相机视野中心
        self.cam_offset_xy = np.array([0.5, 0.0])   # (dx, dy) in robot-base frame，增大前向偏移
        # self.cam_offset_xy = np.array([0.0, 0.0])   # (dx, dy) in robot-base frame，增大前向偏移

        # 将偏移从机器人坐标系转换到世界坐标系
        base_rotation_matrix = t3d_euler.euler2mat(0, 0, yaw)
        offset_world_xy = base_rotation_matrix[:2, :2] @ self.cam_offset_xy
        
        # 计算基座在世界坐标系中的xy位置（允许调整相机-基座水平偏移）
        base_xy = cam_pos[:2] - offset_world_xy
        robot_base_pos = np.array([base_xy[0], base_xy[1], 0.0])   # z坐标固定为0（地面）
        
        # 转换四元数格式：从 [w, x, y, z] 到 [x, y, z, w] (sapien.Pose格式)
        base_quat_sapien = [base_quat[1], base_quat[2], base_quat[3], base_quat[0]]
        head_quat_sapien = [head_quat[1], head_quat[2], head_quat[3], head_quat[0]]

        # 设置机器人基座位置（只考虑yaw旋转，保持机器人直立）
        if self.robot is not None:
            # 设置左臂和右臂机器人基座（它们应该是同一个基座）
            base_pose = sapien.Pose(robot_base_pos, base_quat_sapien)
            self.robot.left_entity.set_root_pose(base_pose)
            self.robot.right_entity.set_root_pose(base_pose)
            
            # 更新机器人的原始姿态记录
            self.robot.left_entity_origion_pose = base_pose
            self.robot.right_entity_origion_pose = base_pose
            
            print(f"Robot base position set to: {robot_base_pos}")

        # 设置相机位置（保持原始的位置和旋转）
        head_pose = sapien.Pose(cam_pos, head_quat_sapien)
        print(f"Setting camera pose: Position {cam_pos}, Quaternion {head_quat_sapien}")
        self.camera.set_pose(head_pose)

        # 若开启第三视角相机，则在机器人基座侧面增加一个相机，朝向机器人基座
        if self.enable_third_person:
            try:
                if self.side_camera is None:
                    self.side_camera = self.scene.add_camera(
                        name="third_person_camera",
                        width=self.image_width,
                        height=self.image_height,
                        fovy=np.deg2rad(self.fovy_deg),  # 使用与主相机相同的视场角
                        near=0.01,
                        far=100.0,  # 增大远平面距离
                    )
                # 使用与第一人称相机相同的位置，但方向相反
                cam_side_pos = cam_pos  # 使用主相机位置
                
                # 创建相反方向的旋转矩阵
                # 从四元数转换为旋转矩阵
                head_quat_wxyz = [head_quat_sapien[3], head_quat_sapien[0], head_quat_sapien[1], head_quat_sapien[2]]  # 从[x,y,z,w]转为[w,x,y,z]
                head_rot_matrix = t3d_quat.quat2mat(head_quat_wxyz)
                
                # 旋转180度（绕Y轴）以获得相反方向
                rot_y_180 = t3d_euler.euler2mat(0, np.pi/6*5, 0)  # 绕Y轴旋转180度
                opposite_rot_matrix = head_rot_matrix @ rot_y_180
                
                # 转回四元数
                opposite_quat_wxyz = t3d_quat.mat2quat(opposite_rot_matrix)
                # 转为Sapien格式 [x,y,z,w]
                side_quat_sapien = [opposite_quat_wxyz[1], opposite_quat_wxyz[2], opposite_quat_wxyz[3], opposite_quat_wxyz[0]]
                
                # 设置第三人称相机位置和朝向
                self.side_camera.set_pose(sapien.Pose(cam_side_pos, side_quat_sapien))
                
                # 打印相机位置和朝向信息
                print(f"第三人称相机位置: {cam_side_pos} (与第一人称相同)")
                print(f"第三人称相机朝向: 与第一人称相反方向(旋转180度)")
                print(f"Third-person camera at {cam_side_pos}, with FOV {self.fovy_deg} degrees")
            except Exception as e:
                print(f"Error setting up third-person camera: {e}")

        self.robot.set_planner(self.scene)  
        print(f"Robot base set to: Position {robot_base_pos}, Yaw {np.rad2deg(yaw):.2f}°")
        
        # 可选：添加调试信息
        print(f"Original rotation matrix:\n{rotation_matrix}")
        print(f"Corrected base rotation (yaw only): {np.rad2deg(yaw):.2f}°")
        print(f"Camera offset in world frame: {offset_world_xy}")
        

    def set_arm_poses(self, 
                     left_end_pos: Union[list, np.ndarray],
                     left_end_rot: Union[list, np.ndarray],
                     right_end_pos: Union[list, np.ndarray], 
                     right_end_rot: Union[list, np.ndarray]):
        """
        设置双臂末端位置和旋转
        
        Args:
            left_end_pos: 左臂末端位置 [x, y, z]
            left_end_rot: 左臂末端旋转 (四元数 [w, x, y, z] 或旋转矩阵 3x3)
            right_end_pos: 右臂末端位置 [x, y, z]
            right_end_rot: 右臂末端旋转 (四元数 [w, x, y, z] 或旋转矩阵 3x3)
        """
        if self.robot is None:
            print("Robot not loaded, cannot set arm poses")
            return
        
        # print(f"3. Setting arm poses - Left: {left_end_pos}, {left_end_rot}, Right: {right_end_pos}, {right_end_rot}")
        try:
            # 处理左臂目标姿态
            left_pos = np.array(left_end_pos).flatten()  # 确保是1D数组
            
            # 处理左臂旋转
            left_end_rot = np.array(left_end_rot)
            if left_end_rot.size == 9:  # 旋转矩阵
                left_rot_matrix = left_end_rot.reshape(3, 3)
                left_quat = t3d_quat.mat2quat(left_rot_matrix) # w,x,y,z
            elif left_end_rot.size == 4:  # 四元数
                left_quat = left_end_rot.flatten()
            else:
                print(f"Invalid left rotation format: {left_end_rot.shape}")
                left_quat = np.array([1, 0, 0, 0])  # 默认四元数
            
            # 处理右臂目标姿态
            right_pos = np.array(right_end_pos).flatten()  # 确保是1D数组
            
            # 处理右臂旋转
            right_end_rot = np.array(right_end_rot)
            if right_end_rot.size == 9:  # 旋转矩阵
                right_rot_matrix = right_end_rot.reshape(3, 3)
                right_quat = t3d_quat.mat2quat(right_rot_matrix)
            elif right_end_rot.size == 4:  # 四元数
                right_quat = right_end_rot.flatten()
            else:
                print(f"Invalid right rotation format: {right_end_rot.shape}")
                right_quat = np.array([1, 0, 0, 0])  # 默认四元数
            
            # 确保四元数格式正确 [w, x, y, z]
            if left_quat.size != 4:
                left_quat = np.array([1, 0, 0, 0])
            if right_quat.size != 4:
                right_quat = np.array([1, 0, 0, 0])
            
            # 构造目标姿态 [x, y, z, qx, qy, qz, qw] (Robot类要求格式)
            left_target_pose = np.concatenate([left_pos[:3], left_quat[1:4], left_quat[0:1]])
            right_target_pose = np.concatenate([right_pos[:3], right_quat[1:4], right_quat[0:1]])
            
            print(f"3. Planning path for left arm to target: {left_target_pose}")
            print(f"4. Planning path for right arm to target: {right_target_pose}")

            # 使用机器人的路径规划功能
            left_plan_result = self.robot.left_plan_path(
                target_pose=left_target_pose.tolist(),
                constraint_pose=None,
                use_point_cloud=False,
                use_attach=False
            )
            
            right_plan_result = self.robot.right_plan_path(
                target_pose=right_target_pose.tolist(),
                constraint_pose=None,
                use_point_cloud=False,
                use_attach=False
            )
            
            print(f"Left planning result: {left_plan_result.get('status', 'Unknown')}")
            print(f"Right planning result: {right_plan_result.get('status', 'Unknown')}")
            
            # 简单安全策略：将z限定到非负，避免落到地面以下
            left_target_pose[2] = max(left_target_pose[2], 0.05)
            right_target_pose[2] = max(right_target_pose[2], 0.05)

            # 检查规划结果
            if (left_plan_result.get('status') == 'Success' and 
                right_plan_result.get('status') == 'Success'):
                print("Path planning successful for both arms")
                
                # 获取规划的路径
                left_path = left_plan_result.get('position', [])
                right_path = right_plan_result.get('position', [])
                
                # 执行到目标位置 (取路径的最后一个点)
                if len(left_path) > 0 and len(right_path) > 0:
                    left_target_joints = left_path[-1]
                    right_target_joints = right_path[-1]
                    
                    print(f"Executing joint positions - Left: {left_target_joints}")
                    print(f"Executing joint positions - Right: {right_target_joints}")
                    
                    # 设置关节目标位置
                    self.robot.set_arm_joints(
                        target_position=left_target_joints,
                        target_velocity=[0.0] * len(left_target_joints),
                        arm_tag="left"
                    )
                    
                    self.robot.set_arm_joints(
                        target_position=right_target_joints,
                        target_velocity=[0.0] * len(right_target_joints),
                        arm_tag="right"
                    )
                    
                    # 运行几步仿真让机器人到达目标位置
                    for _ in range(100):
                        self.scene.step()
                    
                    # 验证最终位置
                    final_left_pose = self.robot.get_left_endpose()
                    final_right_pose = self.robot.get_right_endpose()
                    
                    print(f"Final left end pose: {final_left_pose}")
                    print(f"Final right end pose: {final_right_pose}")
                    
                else:
                    print("Empty path returned from planner")
                    self._set_simplified_arm_poses(left_pos, left_quat, right_pos, right_quat)
            else:
                print(f"Path planning failed - Left: {left_plan_result.get('status', 'Unknown')}, Right: {right_plan_result.get('status', 'Unknown')}")
                # 回退到简单的关节控制
                self._set_simplified_arm_poses(left_pos, left_quat, right_pos, right_quat)
                
        except Exception as e:
            print(f"Error in set_arm_poses: {e}")
            import traceback
            traceback.print_exc()
            # 回退到简化控制
            try:
                self._set_simplified_arm_poses(np.array(left_end_pos), np.array([1,0,0,0]), 
                                             np.array(right_end_pos), np.array([1,0,0,0]))
            except:
                print("Simplified arm poses also failed")
    
    def _set_simplified_arm_poses(self, left_pos, left_quat, right_pos, right_quat):
        print("Setting simplified arm poses (placeholder implementation)???????")
        pass
    
    def _set_arm_ik(self, left_pos, left_quat, right_pos, right_quat):
        """使用逆运动学设置机器人手臂位置"""
        # 这里需要实现具体的逆运动学算法
        # 由于没有具体的机器人模型，这里只是占位符
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
        
        # 获取RGB图像 (去掉alpha通道)
        rgb = camera_rgba_img[:, :, :3]
        
        return rgb

    def render_and_save(self, head_path: str, side_path: Optional[str] = None):
        """渲染并将头部相机和第三视角相机图像保存到文件（若存在）。"""
        # 更新场景与渲染
        self.scene.step()
        self.scene.update_render()

        # 头部相机
        self.camera.take_picture()
        head_rgba = self.camera.get_picture("Color")
        head_img = (head_rgba * 255).clip(0, 255).astype("uint8")[:, :, :3]
        cv2.imwrite(head_path, cv2.cvtColor(head_img, cv2.COLOR_RGB2BGR))
        print(f"Head camera image saved to: {head_path}")

        # 第三视角相机（如启用且提供保存路径）
        if self.side_camera is not None and side_path is not None:
            self.side_camera.take_picture()
            side_rgba = self.side_camera.get_picture("Color")
            side_img = (side_rgba * 255).clip(0, 255).astype("uint8")[:, :, :3]
            cv2.imwrite(side_path, cv2.cvtColor(side_img, cv2.COLOR_RGB2BGR))
            print(f"Third-person camera image saved to: {side_path}")
    
    def render_scene(self, 
                    camera_extrinsics: Dict[str, Any],
                    left_end_pos: Union[list, np.ndarray],
                    left_end_rot: Union[list, np.ndarray],
                    right_end_pos: Union[list, np.ndarray],
                    right_end_rot: Union[list, np.ndarray]) -> np.ndarray:
        """
        一键渲染：设置所有参数并渲染场景
        
        Args:
            camera_extrinsics: 相机外参
            left_end_pos: 左臂末端位置
            left_end_rot: 左臂末端旋转
            right_end_pos: 右臂末端位置  
            right_end_rot: 右臂末端旋转
            
        Returns:
            np.ndarray: 渲染的RGB图像
        """
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


def demo_usage():
    """从 clean_cups/0_wrist_data.json 读取第0帧外参与左右手姿态并渲染一帧"""
    import numpy as np
    import json

    json_path = "/data1/zjyang/program/third/RoboTwin/code_painting/clean_cups/0_wrist_data.json"
    with open(json_path, "r") as f:
        data = json.load(f)

    # 读取相机内参（如果存在）
    if "intrinsics" in data["camera"]:
        camera_intrinsics = np.array(data["camera"]["intrinsics"], dtype=float)
        print(f"读取到相机内参矩阵:\n{camera_intrinsics}")
        
        # 从内参矩阵提取焦距和主点
        fx = camera_intrinsics[0, 0]
        fy = camera_intrinsics[1, 1]
        cx = camera_intrinsics[0, 2]
        cy = camera_intrinsics[1, 2]
        
        # 计算视场角（垂直方向）
        image_height = data["camera"].get("height", 480)  # 默认高度为480
        fovy_rad = 2 * np.arctan(image_height / (2 * fy))
        fovy_deg = np.rad2deg(fovy_rad)
        print(f"从内参计算的视场角: {fovy_deg:.2f}°")
    else:
        print("JSON中未找到相机内参，使用默认视场角")
        fovy_deg = 90.0  # 默认视场角

    # 读取第0帧相机外参（OpenCV坐标系的4x4外参矩阵）
    T_cv = np.array(data["camera"]["transforms"][0], dtype=float)  # world_from_camera (OpenCV basis)
    print(f"原始相机外参矩阵 T_cv:\n{T_cv}")

    # 1) 世界系基底变换：W0(原始) -> Wt(目标: Z↑, X前, Y右)
    R_wt_from_w0 = np.array([[0, 1, 0],
                            [-1, 0, 0],
                            [0, 0, 1]], dtype=float)
    T_cv_wt = T_cv.copy()
    T_cv_wt[:3, :3] = R_wt_from_w0 @ T_cv[:3, :3]
    T_cv_wt[:3,  3] = R_wt_from_w0 @ T_cv[:3,  3]

    # OpenCV -> OpenGL 轴系转换矩阵（等价于绕X轴旋转π）
    cv_to_gl = np.array([
        [1,  0,  0, 0],
        [0, -1,  0, 0],
        [0,  0, -1, 0],
        [0,  0,  0, 1],
    ], dtype=float)

    # 得到用于渲染/Viewer 的 OpenGL 相机外参
    # T_gl = T_cv @ cv_to_gl
    T_gl_wt = T_cv_wt @ cv_to_gl
    # print(f"转换后的相机外参矩阵 T_gl:\n{T_gl}")
    
    # 提取相机位置和旋转
    # camera_position = T_gl[:3, 3]
    # camera_rotation = T_gl[:3, :3]
    camera_position = T_gl_wt[:3, 3]
    camera_rotation = T_gl_wt[:3, :3]   
    
    # 调整相机位置，使其更靠近机器人，并略微提高高度
    # 这有助于确保机器人在视野内
    camera_position[2] += 0.3  # 提高相机高度
    
    camera_extrinsics = {"position": camera_position.tolist(), "rotation": camera_rotation.tolist()}
    print(f"Camera position: {camera_position}")
    print(f"Camera rotation matrix:\n{camera_rotation}")

    # 读取左右手第0帧末端位姿（原始在相机坐标系，OpenCV basis）
    left_pos_cam_cv = np.array(data["wrists"]["left"]["position"][0], dtype=float)
    left_rot_cam_cv = np.array(data["wrists"]["left"]["orientation"][0], dtype=float)
    right_pos_cam_cv = np.array(data["wrists"]["right"]["position"][0], dtype=float)
    right_rot_cam_cv = np.array(data["wrists"]["right"]["orientation"][0], dtype=float)

    # 打印转换后的末端位置和旋转
    print("\n===== 坐标系和位姿信息 =====")
    print("【位置坐标】: [x, y, z] - 表示三维空间中的位置")
    print("【旋转表示】: 四元数 [w, x, y, z] - 表示三维空间中的旋转")
    print("【相机位置】: 相机在世界坐标系中的位置 camera_position")
    print("【相机朝向】: 相机的旋转矩阵 camera_rotation 或四元数表示")
    print("【机器人手臂】: 左右手臂的末端位置和旋转")
    print("\n----- 具体数值 -----")
    print(f"相机位置: {camera_position}")
    print(f"相机旋转矩阵:\n{camera_rotation}")
    # 仅打印原始相机系下手腕信息；实际转换在 set_arm_poses_from_camera_frame 内完成
    print(f"相机系左腕位置: {left_pos_cam_cv}")
    print(f"相机系右腕位置: {right_pos_cam_cv}")
    
    # 获取图像尺寸（如果在JSON中有指定）
    image_width = data["camera"].get("width", 800)  # 默认宽度为800
    image_height = data["camera"].get("height", 600)  # 默认高度为600
    
    # 初始化渲染器（使用从内参计算的视场角）
    renderer = RobotRenderer(
        image_width=image_width,
        image_height=image_height,
        enable_viewer=True,
        fovy_deg=fovy_deg,  # 使用从内参计算的视场角
        enable_third_person=True,
        third_person_distance=1.5,  # 减小距离，更靠近机器人
        third_person_height=1.2,
        third_person_yaw_offset_deg=90.0,  # 调整为侧视角
        third_person_pitch_deg=-30.0,  # 向下看30度
    )

    try:
        # 设置机器人基座与相机
        renderer.set_robot_base_pose(camera_extrinsics)

        # 根据“机械臂坐标 × 相机坐标系”从相机系手腕位姿计算世界位姿并设置
        renderer.set_arm_poses_from_camera_frame(
            T_world_from_camera_cv=T_cv,
            left_pos_cam_cv=left_pos_cam_cv,
            left_rot_cam_cv=left_rot_cam_cv,
            right_pos_cam_cv=right_pos_cam_cv,
            right_rot_cam_cv=right_rot_cam_cv,
        )

        # 渲染图像
        rgb_image = renderer.render_frame()

        # 保存图像：无头运行，保存头部与第三视角
        renderer.save_image(rgb_image, "robot_render_result11.png")
        # 再次调用统一的双相机保存接口，保证第三视角也落盘
        renderer.render_and_save(head_path="code_painting/robot_render_result11.png", \
         side_path="code_painting/robot_render_result11_side.png")
        print("渲染结果已保存到 robot_render_result10.png 和 robot_render_result10_side.png")
        
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



if __name__ == "__main__":
    demo_usage()