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
                 enable_viewer: bool = False):
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
            fovy=np.deg2rad(45),  # 35度视场角
            near=0.1,
            far=10
        )
        
        # 设置默认相机位置 (头部相机)
        self.camera.set_entity_pose(sapien.Pose([0, 0, 1.0], [0.7071, 0, 0.7071, 0]))
    
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
        # 相机的前向向量（通常是-z方向或+z方向，取决于坐标系定义）
        camera_forward = rotation_matrix[:, 2]  # 或者 rotation_matrix[:, 0] 取决于相机坐标系定义
        
        # 将前向向量投影到xy平面，计算yaw角
        forward_xy = camera_forward[:2]  # 只取x,y分量
        forward_xy_normalized = forward_xy / (np.linalg.norm(forward_xy) + 1e-8)  # 避免除零
        
        # 计算yaw角（相对于x轴的角度）
        yaw = np.arctan2(forward_xy_normalized[1], forward_xy_normalized[0])
        
        # 如果您的相机坐标系定义不同，可能需要调整角度
        # 例如：yaw = yaw + np.pi  # 如果需要180度偏移
        # 或者：yaw = -yaw  # 如果需要反向
        
        print(f"Camera forward vector: {camera_forward}")
        print(f"Extracted yaw angle: {np.rad2deg(yaw):.2f} degrees")
        
        # 创建只包含yaw旋转的四元数（保持机器人直立）
        base_quat = t3d_euler.euler2quat(0, 0, yaw)  # roll=0, pitch=0, 只有yaw
        
        # 对于相机头部，我们保持原始的旋转
        head_quat = t3d_quat.mat2quat(rotation_matrix)
        
        # 计算机器人基座位置
        # 假设相机在机器人头部，需要考虑相机到基座的偏移
        self.cam_offset_xy = np.array([0.2, 0.0])   # (dx, dy) in robot-base frame
        
        # 将偏移从机器人坐标系转换到世界坐标系
        base_rotation_matrix = t3d_euler.euler2mat(0, 0, yaw)
        offset_world_xy = base_rotation_matrix[:2, :2] @ self.cam_offset_xy
        
        # 计算基座在世界坐标系中的xy位置
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

        # 设置相机位置（保持原始的位置和旋转）
        head_pose = sapien.Pose(cam_pos, head_quat_sapien)
        print(f"Setting camera pose: Position {cam_pos}, Quaternion {head_quat_sapien}")
        
        self.camera.set_pose(head_pose)
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
    """使用真实外参和左右手姿态渲染场景"""
    import numpy as np

    # 解析 camera 外参 4x4 矩阵 OpenCV 格式
    # camera_transform = np.array([
    #     [0.9499143958091736, 0.04421062394976616, -0.3093675673007965, -0.036132317036390305],
    #     [-0.08030540496110916, -0.9221678972244263, -0.37836185097694397, 1.2202030420303345],
    #     [-0.30201631784439087, 0.3842552900314331, -0.8724300265312195, 1.22493996560573578],
    #     [0.0, 0.0, 0.0, 1.0000001192092896]
    # ])
    
    camera_transform = np.array([
        [0.9461427927017212, 0.14764580130577087, -0.2881225049495697, -0.12369639426469803],
        [0.09595535695552826, -0.9778545498847961, -0.18599244952201843, 1.032541275024414],
        [-0.3092028796672821, 0.14832855761051178, -0.939357340335846, 1.2420856451392174],
        [0.0, 0.0, 0.0, 1.0]
    ])
    
    # 直接将外参转为OpenGL格式
    # ---------------- 修正旋转 -------------------
    R_fix   = t3d_euler.euler2mat(np.pi, 0, 0)     # rotX(π)
    R_cv    = camera_transform[:3, :3]
    R_gl    = R_cv @ R_fix                         # ← 左乘也行；右手系保持
    camera_transform[:3, :3] = R_gl
    
    # ---------------------------------------------
    camera_position = camera_transform[:3, 3]
    camera_rotation = camera_transform[:3, :3]

    camera_extrinsics = {
        'position': camera_position.tolist(),
        'rotation': camera_rotation.tolist()
    }

    # 右手位置 & 旋转矩阵
    # right_end_pos = [-0.13050371408462524, 1.1453771591186523, 0.97134928345680237]
    # right_end_rot = [
    #     [0.44208455085754395, 0.8873743414878845, -0.13087667524814606],
    #     [0.46062690019607544, -0.3497961163520813, -0.8157610893249512],
    #     [-0.7696652412414551, 0.30035001039505005, -0.5633876919746399]
    # ]
    
    # right_end_pos = [-0.27100253105163574, 1.0040560960769653, -0.35425981879234314]
    right_end_pos = [-0.27100253105163574, 1.0040560960769653, 1.01425981879234314]

    right_end_rot = [
        [0.526455283164978, 0.3962542712688446, -0.7522145509719849],
        [0.4936225116252899, -0.862812876701355, -0.10904219001531601],
        [-0.6922289133071899, -0.3139042556285858, -0.6498324871063232]
    ]

    # 左手位置 & 旋转矩阵
    # left_end_pos = [0.15974220633506775, 1.0882387161254883, 0.8029957294464111]
    # left_end_rot = [
    #     [0.2292477786540985, 0.9499625563621521, -0.2121734768152237],
    #     [-0.4682941138744354, 0.2987341582775116, 0.8315404057502747],
    #     [0.8533154726028442, -0.09126907587051392, 0.5133456587791443]
    # ]
    left_end_pos = [-0.06912250071763992, 0.9271786212921143, 0.9354623258113861]
    # left_end_pos = [-0.06912250071763992, 0.9271786212921143, -0.4354623258113861]

    left_end_rot = [
        [0.5236368179321289, 0.5102211833000183, -0.6822596192359924],
        [-0.7503378391265869, 0.6554761528968811, -0.08569566160440445],
        [0.403481125831604, 0.5567986965179443, 0.7260698676109314]
    ]

    # 初始化渲染器
    renderer = RobotRenderer(image_width=640, image_height=360, enable_viewer=True)

    try:
        # 渲染图像
        rgb_image = renderer.render_scene(
            camera_extrinsics=camera_extrinsics,
            left_end_pos=left_end_pos,
            left_end_rot=left_end_rot,
            right_end_pos=right_end_pos,
            right_end_rot=right_end_rot
        )

        # 保存图像
        renderer.save_image(rgb_image, "robot_render_result1.png")
        
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