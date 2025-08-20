

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

def demo_usage():
    """使用正确转换的外参和左右手姿态渲染场景"""
    import numpy as np

    # 原始OpenCV格式的相机外参
    camera_transform_opencv = np.array([
        [0.9461427927017212, 0.14764580130577087, -0.2881225049495697, -0.12369639426469803],
        [0.09595535695552826, -0.9778545498847961, -0.18599244952201843, 1.032541275024414],
        [-0.3092028796672821, 0.14832855761051178, -0.939357340335846, 1.2420856451392174],
        [0.0, 0.0, 0.0, 1.0]
    ])
    
    print("原始OpenCV外参:")
    print(camera_transform_opencv)
    print()
    
    # 正确转换为OpenGL格式
    def convert_opencv_to_opengl(cv_transform):
        """OpenCV到OpenGL的正确转换"""
        # OpenCV到OpenGL的转换矩阵
        cv_to_gl = np.array([
            [1,  0,  0,  0],
            [0, -1,  0,  0],  # Y轴翻转（下变上）
            [0,  0, -1,  0],  # Z轴翻转（前变后）
            [0,  0,  0,  1]
        ])
        
        # 正确的转换方式
        gl_transform = cv_transform @ cv_to_gl
        return gl_transform
    
    # 转换为OpenGL格式
    camera_transform_opengl = convert_opencv_to_opengl(camera_transform_opencv)
    
    print("转换后的OpenGL外参:")
    print(camera_transform_opengl)
    print()
    
    # 提取位置和旋转
    camera_position = camera_transform_opengl[:3, 3]
    camera_rotation = camera_transform_opengl[:3, :3]

    camera_extrinsics = {
        'position': camera_position.tolist(),
        'rotation': camera_rotation.tolist()
    }
    
    print("相机位置:", camera_position)
    print("相机旋转矩阵:")
    print(camera_rotation)
    print()

    # 右手位置 & 旋转矩阵
    right_end_pos = [-0.27100253105163574, 1.0040560960769653, 0.98425981879234314]
    right_end_rot = [
        [0.526455283164978, 0.3962542712688446, -0.7522145509719849],
        [0.4936225116252899, -0.862812876701355, -0.10904219001531601],
        [-0.6922289133071899, -0.3139042556285858, -0.6498324871063232]
    ]

    # 左手位置 & 旋转矩阵
    left_end_pos = [-0.06912250071763992, 0.9271786212921143, 0.9054623258113861]
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
        renderer.save_image(rgb_image, "robot_render_result_corrected.png")
        
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
    
    print("Rendering complete with corrected camera extrinsics.")

# 用于验证转换正确性的辅助函数
def verify_conversion():
    """验证OpenCV到OpenGL转换的正确性"""
    
    # 测试：相机在原点，看向+Z方向（OpenCV）
    opencv_looking_forward = np.array([
        [1, 0, 0, 0],  # X轴不变
        [0, 1, 0, 0],  # Y轴向下
        [0, 0, 1, 0],  # Z轴向前
        [0, 0, 0, 1]
    ])
    
    # 转换为OpenGL
    cv_to_gl = np.array([
        [1,  0,  0,  0],
        [0, -1,  0,  0],
        [0,  0, -1,  0],
        [0,  0,  0,  1]
    ])
    
    opengl_result = opencv_looking_forward @ cv_to_gl
    
    print("OpenCV相机看向+Z方向时的变换矩阵:")
    print(opencv_looking_forward)
    print("\n转换为OpenGL后（应该看向-Z方向）:")
    print(opengl_result)
    
    # 预期结果：OpenGL中相机看向-Z方向
    expected_opengl = np.array([
        [1,  0,  0, 0],  # X轴不变
        [0, -1,  0, 0],  # Y轴向上
        [0,  0, -1, 0],  # Z轴向后
        [0,  0,  0, 1]
    ])
    
    print("\n预期的OpenGL结果:")
    print(expected_opengl)
    print("\n差异:", np.linalg.norm(opengl_result - expected_opengl))

if __name__ == "__main__":
    print("=== 验证转换正确性 ===")
    verify_conversion()
    print("\n" + "="*50 + "\n")
    
    print("=== 运行修正后的渲染 ===")
    demo_usage()