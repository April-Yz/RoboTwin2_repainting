#!/usr/bin/env python3
import json
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import os

def load_json_file(filepath):
    """安全加载JSON文件"""
    try:
        if not os.path.exists(filepath):
            print(f"错误: 文件 {filepath} 不存在")
            return None
        
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"错误: 读取文件 {filepath} 失败: {e}")
        return None

# def extract_camera_positions(aria_data):
#     """从相机外参中提取相机位置"""
#     if not aria_data:
#         return np.array([])
    
#     # camera_extrinsics = aria_data.get('aria01_content', {}).get('camera_extrinsics', {})
#     camera_extrinsics = aria_data.get('camera', {}).get('transforms', {})

#     positions = []
    
#     # 按时间戳顺序排序
#     sorted_timestamps = sorted(camera_extrinsics.keys(), key=lambda x: int(x))
    
#     for timestamp in sorted_timestamps:
#         # 外参矩阵是4x3的形式，我们需要转换为4x4矩阵
#         extrinsic = camera_extrinsics[timestamp]
#         if len(extrinsic) == 3 and len(extrinsic[0]) == 4:
#             # 提取平移部分 (最后一列的前三个元素)
#             position = [extrinsic[0][3], extrinsic[1][3], extrinsic[2][3]]
#             positions.append(position)
    
#     return np.array(positions)

def extract_camera_positions_combined(combined_data):
    """从组合数据文件中提取相机位置"""
    if not combined_data:
        return np.array([])
    
    camera_transforms = combined_data.get('camera', {}).get('transforms', [])
    positions = []
    
    for transform in camera_transforms:
        if len(transform) == 4 and len(transform[0]) == 4:
            # 提取平移部分 (最后一列的前三个元素)
            position = [transform[0][3], transform[1][3], transform[2][3]]
            positions.append(position)
    
    return np.array(positions)

def extract_camera_orientations(combined_data):
    """从组合数据文件中提取相机旋转矩阵"""
    if not combined_data:
        return np.array([])

    camera_transforms = combined_data.get('camera', {}).get('transforms', [])
    rotations = []

    for transform in camera_transforms:
        if len(transform) == 4 and len(transform[0]) == 4:
            # 提取旋转矩阵部分 (前三行前三列)
            rotation_matrix = [
                [transform[0][0], transform[0][1], transform[0][2]],
                [transform[1][0], transform[1][1], transform[1][2]],
                [transform[2][0], transform[2][1], transform[2][2]],
            ]
            rotations.append(rotation_matrix)

    return np.array(rotations)

def extract_wrist_positions(wrist_data, max_points=None):
    """从手腕数据中提取位置信息"""
    if not wrist_data:
        return np.array([]), np.array([])
    
    # trajectories = wrist_data.get('trajectories', [])
    trajectories = wrist_data.get('wrists', [])
    if not trajectories:
        return np.array([]), np.array([])
    
    sample = trajectories[0]
    
    
    # 提取左手腕位置
    left_wrist = sample.get('left', {})
    left_positions = np.array(left_wrist.get('positions', []))
    
    # 提取右手腕位置
    right_wrist = sample.get('right', {})
    right_positions = np.array(right_wrist.get('positions', []))
    
    # 如果指定了最大点数，则截取
    if max_points is not None:
        left_positions = left_positions[:max_points]
        right_positions = right_positions[:max_points]
    
    return left_positions, right_positions


def extract_wrist_positions_combined(wrist_data, max_points=None):
    """从手腕数据中提取位置信息"""
    if not wrist_data:
        return np.array([]), np.array([])

    wrists = wrist_data.get('wrists', {})
    if not wrists:
        return np.array([]), np.array([])

    # 提取左手腕位置
    left_wrist = wrists.get('left', {})
    left_positions = np.array(left_wrist.get('position', []))  # ✅ 注意是 'position'

    # 提取右手腕位置
    right_wrist = wrists.get('right', {})
    right_positions = np.array(right_wrist.get('position', []))  # ✅ 注意是 'position'

    # 如果指定了最大点数，则截取
    if max_points is not None:
        left_positions = left_positions[:max_points]
        right_positions = right_positions[:max_points]

    return left_positions, right_positions

def extract_wrist_orientation_combined(wrist_data, max_points=None):
    """从手腕数据中提取朝向信息"""
    if not wrist_data:
        return np.array([]), np.array([])

    wrists = wrist_data.get('wrists', {})
    if not wrists:
        return np.array([]), np.array([])

    # 提取左手腕朝向
    left_wrist = wrists.get('left', {})
    left_orientations = np.array(left_wrist.get('orientation', []))  # ✅ 注意是 'orientation'

    # 提取右手腕朝向
    right_wrist = wrists.get('right', {})
    right_orientations = np.array(right_wrist.get('orientation', []))  # ✅ 注意是 'orientation'
    
    # 打印调试信息
    print(f"左手朝向数据形状: {left_orientations.shape}")
    print(f"右手朝向数据形状: {right_orientations.shape}")
    
    # 检查是否有空数据
    if len(left_orientations) == 0:
        print("警告：左手朝向数据为空")
    if len(right_orientations) == 0:
        print("警告：右手朝向数据为空")
    
    # 检查数据是否有效
    if len(left_orientations) > 0:
        print(f"左手朝向数据样例: {left_orientations[0]}")
    if len(right_orientations) > 0:
        print(f"右手朝向数据样例: {right_orientations[0]}")

    # 如果指定了最大点数，则截取
    if max_points is not None:
        left_orientations = left_orientations[:max_points]
        right_orientations = right_orientations[:max_points]

    return left_orientations, right_orientations

def plot_2d_trajectories(camera_pos, left_pos, right_pos):
    """绘制2D俯视图"""
    
    # 创建图形和子图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # === 左侧图：完整轨迹 ===
    ax1.set_title('Camera and Wrist Trajectories (Top View)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('X (m)', fontsize=12)
    ax1.set_ylabel('Y (m)', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal', adjustable='box')
    
    # 绘制相机轨迹（黑色）
    if len(camera_pos) > 0:
        ax1.plot(camera_pos[:, 0], camera_pos[:, 1], 'k-', linewidth=2, label='Camera Trajectory', alpha=0.8)
        ax1.scatter(camera_pos[0, 0], camera_pos[0, 1], c='gray', s=100, marker='o', 
                   label='Camera Start Point', zorder=5, edgecolors='black', linewidth=1)
        ax1.scatter(camera_pos[-1, 0], camera_pos[-1, 1], c='black', s=100, marker='s', 
                   label='Camera End Point', zorder=5, edgecolors='white', linewidth=1)
    
    # 绘制左手腕轨迹（蓝色）
    if len(left_pos) > 0:
        ax1.plot(left_pos[:, 0], left_pos[:, 1], 'b-', linewidth=2, label='Left Wrist Trajectory', alpha=0.8)
        ax1.scatter(left_pos[0, 0], left_pos[0, 1], c='lightblue', s=80, marker='o', 
                   label='Left Wrist Start Point', zorder=5, edgecolors='blue', linewidth=1)
        ax1.scatter(left_pos[-1, 0], left_pos[-1, 1], c='blue', s=80, marker='s', 
                   label='Left Wrist End Point', zorder=5, edgecolors='white', linewidth=1)
    
    # 绘制右手腕轨迹（绿色）
    if len(right_pos) > 0:
        ax1.plot(right_pos[:, 0], right_pos[:, 1], 'g-', linewidth=2, label='Right Wrist Trajectory', alpha=0.8)
        ax1.scatter(right_pos[0, 0], right_pos[0, 1], c='lightgreen', s=80, marker='o', 
                   label='Right wrist start point', zorder=5, edgecolors='green', linewidth=1)
        ax1.scatter(right_pos[-1, 0], right_pos[-1, 1], c='green', s=80, marker='s', 
                   label='Right wrist end point', zorder=5, edgecolors='white', linewidth=1)
    
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # === 右侧图：时间演化（颜色渐变） ===
    ax2.set_title('Trajectory Time Evolution (Color Indicates Time)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('X (m)', fontsize=12)
    ax2.set_ylabel('Y (m)', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal', adjustable='box')
    
    # 使用颜色映射显示时间演化
    min_len = min(len(camera_pos), len(left_pos), len(right_pos))
    if min_len > 0:
        time_indices = np.arange(min_len)
        
        # 相机轨迹（灰度渐变）
        scatter_cam = ax2.scatter(camera_pos[:min_len, 0], camera_pos[:min_len, 1], 
                                 c=time_indices, cmap='Greys', s=20, alpha=0.7, label='Camera')
        
        # 左手腕轨迹（蓝色渐变）
        scatter_left = ax2.scatter(left_pos[:min_len, 0], left_pos[:min_len, 1], 
                                  c=time_indices, cmap='Blues', s=15, alpha=0.7, label='Left Wrist')
        
        # 右手腕轨迹（绿色渐变）
        scatter_right = ax2.scatter(right_pos[:min_len, 0], right_pos[:min_len, 1], 
                                   c=time_indices, cmap='Greens', s=15, alpha=0.7, label='Right Wrist')
        
        # 添加颜色条
        cbar = plt.colorbar(scatter_cam, ax=ax2, shrink=0.8)
        cbar.set_label('Timesteps', fontsize=10)
        
        # 标记起始点
        ax2.scatter(camera_pos[0, 0], camera_pos[0, 1], c='red', s=100, marker='*', 
                   label='start_point', zorder=10, edgecolors='black', linewidth=1)
    
    ax2.legend()
    
    # 调整布局
    plt.tight_layout()
    
    # 显示统计信息
    print("\n=== 2D轨迹统计信息 ===")
    if len(camera_pos) > 0:
        x_range = np.max(camera_pos[:, 0]) - np.min(camera_pos[:, 0])
        y_range = np.max(camera_pos[:, 1]) - np.min(camera_pos[:, 1])
        print(f"相机XY平面移动范围: X={x_range:.3f}m, Y={y_range:.3f}m")
        print(f"相机起始位置: ({camera_pos[0, 0]:.3f}, {camera_pos[0, 1]:.3f})")
        print(f"相机结束位置: ({camera_pos[-1, 0]:.3f}, {camera_pos[-1, 1]:.3f})")
    
    if len(left_pos) > 0:
        x_range = np.max(left_pos[:, 0]) - np.min(left_pos[:, 0])
        y_range = np.max(left_pos[:, 1]) - np.min(left_pos[:, 1])
        print(f"左手腕XY平面移动范围: X={x_range:.3f}m, Y={y_range:.3f}m")
        print(f"左手腕起始位置: ({left_pos[0, 0]:.3f}, {left_pos[0, 1]:.3f})")
        print(f"左手腕结束位置: ({left_pos[-1, 0]:.3f}, {left_pos[-1, 1]:.3f})")
    
    if len(right_pos) > 0:
        x_range = np.max(right_pos[:, 0]) - np.min(right_pos[:, 0])
        y_range = np.max(right_pos[:, 1]) - np.min(right_pos[:, 1])
        print(f"右手腕XY平面移动范围: X={x_range:.3f}m, Y={y_range:.3f}m")
        print(f"右手腕起始位置: ({right_pos[0, 0]:.3f}, {right_pos[0, 1]:.3f})")
        print(f"右手腕结束位置: ({right_pos[-1, 0]:.3f}, {right_pos[-1, 1]:.3f})")
    
    # 计算相对位置关系
    if len(camera_pos) > 0 and len(left_pos) > 0 and len(right_pos) > 0:
        print(f"\n=== 相对位置关系分析 ===")
        # 计算平均位置
        cam_center = np.mean(camera_pos[:min_len], axis=0)
        left_center = np.mean(left_pos[:min_len], axis=0)
        right_center = np.mean(right_pos[:min_len], axis=0)
        
        print(f"相机平均位置: ({cam_center[0]:.3f}, {cam_center[1]:.3f})")
        print(f"左手腕平均位置: ({left_center[0]:.3f}, {left_center[1]:.3f})")
        print(f"右手腕平均位置: ({right_center[0]:.3f}, {right_center[1]:.3f})")
        
        # 计算距离
        cam_left_dist = np.linalg.norm(cam_center[:2] - left_center[:2])
        cam_right_dist = np.linalg.norm(cam_center[:2] - right_center[:2])
        left_right_dist = np.linalg.norm(left_center[:2] - right_center[:2])
        
        print(f"相机到左手腕平均距离: {cam_left_dist:.3f}m")
        print(f"相机到右手腕平均距离: {cam_right_dist:.3f}m")
        print(f"左右手腕间平均距离: {left_right_dist:.3f}m")


def plot_3d_trajectories(camera_positions, left_wrist_positions, left_wrist_orientations, right_wrist_positions, right_wrist_orientations, camera_orientations=None, skip=20):
    """绘制 3D 轨迹图 + 相机朝向箭头"""
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 摄像头轨迹
    if len(camera_positions) > 0:
        ax.plot(camera_positions[:, 0], camera_positions[:, 1], camera_positions[:, 2], label='Camera', color='blue')
    # 左手腕轨迹
    if len(left_wrist_positions) > 0:
        ax.plot(left_wrist_positions[:, 0], left_wrist_positions[:, 1], left_wrist_positions[:, 2], label='Left Wrist', color='green')
    # 右手腕轨迹
    if len(right_wrist_positions) > 0:
        ax.plot(right_wrist_positions[:, 0], right_wrist_positions[:, 1], right_wrist_positions[:, 2], label='Right Wrist', color='red')

    # 绘制相机朝向
    # if camera_orientations is not None:
    #     for i in range(0, len(camera_positions), skip):
    #         origin = camera_positions[i]
    #         orientation = camera_orientations[i]
    #         print(f"Camera position {i}: {origin}, Orientation: {orientation}")
    #         breakpoint()
            
    #         # 绘制三个坐标轴方向
    #         ax.quiver(origin[0], origin[1], origin[2],
    #                   orientation[0][0], orientation[1][0], orientation[2][0],
    #                   length=0.1, color='red', normalize=True, label='X-axis' if i == 0 else "")
    #         ax.quiver(origin[0], origin[1], origin[2],
    #                   orientation[0][1], orientation[1][1], orientation[2][1],
    #                   length=0.1, color='green', normalize=True, label='Y-axis' if i == 0 else "")
    #         ax.quiver(origin[0], origin[1], origin[2],
    #                   orientation[0][2], orientation[1][2], orientation[2][2],
    #                   length=0.1, color='blue', normalize=True, label='Z-axis' if i == 0 else "")
    
    def draw_orientation_arrow(origin, orientation, color_set, label_prefix="", i=0):
        """绘制以 origin 为起点的方向箭头，orientation 为 3x3 旋转矩阵"""
        orientation = np.array(orientation)
        if orientation.shape == (9,):
            orientation = orientation.reshape(3, 3)
        elif orientation.shape != (3, 3):
            print(f"Skip invalid orientation shape: {orientation.shape}")
            return

        ax.quiver(*origin, *orientation[:, 0], length=0.05, color=color_set[0], normalize=True, label=f'{label_prefix} X-axis' if i == 0 else "")
        ax.quiver(*origin, *orientation[:, 1], length=0.05, color=color_set[1], normalize=True, label=f'{label_prefix} Y-axis' if i == 0 else "")
        ax.quiver(*origin, *orientation[:, 2], length=0.05, color=color_set[2], normalize=True, label=f'{label_prefix} Z-axis' if i == 0 else "")

    # 相机朝向
    if camera_orientations is not None:
        for i in range(0, len(camera_positions), skip):
            draw_orientation_arrow(camera_positions[i], camera_orientations[i], ['red', 'green', 'blue'], label_prefix='Cam', i=i)

    # 左手朝向
    if left_wrist_orientations is not None:
        for i in range(0, len(left_wrist_positions), skip):
            draw_orientation_arrow(left_wrist_positions[i], left_wrist_orientations[i], ['darkred', 'darkgreen', 'navy'], label_prefix='L-Wrist', i=i)

    # 右手朝向
    if right_wrist_orientations is not None:
        for i in range(0, len(right_wrist_positions), skip):
            draw_orientation_arrow(right_wrist_positions[i], right_wrist_orientations[i], ['orangered', 'lime', 'skyblue'], label_prefix='R-Wrist', i=i)


    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title("3D Trajectories with Camera Orientation")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.show()
    
def create_camera_visualization(positions, color=[0, 0, 0]):
    """创建相机位置的可视化"""
    if len(positions) == 0:
        return []
    
    geometries = []
    
    # 创建相机轨迹线
    if len(positions) > 1:
        lines = [[i, i+1] for i in range(len(positions)-1)]
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(positions)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.colors = o3d.utility.Vector3dVector([color for _ in lines])
        geometries.append(line_set)
    
    # 创建相机位置点云
    camera_pcd = o3d.geometry.PointCloud()
    camera_pcd.points = o3d.utility.Vector3dVector(positions)
    camera_pcd.colors = o3d.utility.Vector3dVector([color for _ in positions])
    geometries.append(camera_pcd)
    
    # 在起始和结束位置添加较大的标记球
    if len(positions) > 0:
        # 起始位置 (稍大的球)
        start_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.02)
        start_sphere.translate(positions[0])
        start_sphere.paint_uniform_color([0.8, 0.8, 0.8])  # 浅灰色
        geometries.append(start_sphere)
        
        # 结束位置 (稍大的球)
        end_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.02)
        end_sphere.translate(positions[-1])
        end_sphere.paint_uniform_color([0.3, 0.3, 0.3])  # 深灰色
        geometries.append(end_sphere)
    
    return geometries

def create_wrist_visualization(positions, color, name):
    """创建手腕轨迹的可视化"""
    if len(positions) == 0:
        return []
    
    geometries = []
    
    # 创建轨迹线
    if len(positions) > 1:
        lines = [[i, i+1] for i in range(len(positions)-1)]
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(positions)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.colors = o3d.utility.Vector3dVector([color for _ in lines])
        geometries.append(line_set)
    
    # 创建轨迹点云（每10个点显示一个，避免太密集）
    sample_indices = range(0, len(positions), 10)
    sampled_positions = positions[sample_indices]
    
    wrist_pcd = o3d.geometry.PointCloud()
    wrist_pcd.points = o3d.utility.Vector3dVector(sampled_positions)
    wrist_pcd.colors = o3d.utility.Vector3dVector([color for _ in sampled_positions])
    geometries.append(wrist_pcd)
    
    # 在起始和结束位置添加标记球
    if len(positions) > 0:
        # 起始位置
        start_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.015)
        start_sphere.translate(positions[0])
        start_sphere.paint_uniform_color([c*0.7 for c in color])  # 稍暗的颜色
        geometries.append(start_sphere)
        
        # 结束位置
        end_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.015)
        end_sphere.translate(positions[-1])
        end_sphere.paint_uniform_color([c*1.3 if c*1.3 <= 1 else 1 for c in color])  # 稍亮的颜色
        geometries.append(end_sphere)
    
    return geometries

def add_coordinate_frame():
    """添加坐标系"""
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
    return coordinate_frame

def print_statistics(camera_pos, left_pos, right_pos):
    """打印统计信息"""
    print("\n=== 可视化数据统计 ===")
    print(f"相机位置数据点: {len(camera_pos)}")
    print(f"左手腕位置数据点: {len(left_pos)}")
    print(f"右手腕位置数据点: {len(right_pos)}")
    
    if len(camera_pos) > 0:
        cam_range = np.max(camera_pos, axis=0) - np.min(camera_pos, axis=0)
        print(f"相机移动范围: X={cam_range[0]:.3f}, Y={cam_range[1]:.3f}, Z={cam_range[2]:.3f}")
    
    if len(left_pos) > 0:
        left_range = np.max(left_pos, axis=0) - np.min(left_pos, axis=0)
        print(f"左手腕移动范围: X={left_range[0]:.3f}, Y={left_range[1]:.3f}, Z={left_range[2]:.3f}")
    
    if len(right_pos) > 0:
        right_range = np.max(right_pos, axis=0) - np.min(right_pos, axis=0)
        print(f"右手腕移动范围: X={right_range[0]:.3f}, Y={right_range[1]:.3f}, Z={right_range[2]:.3f}")

def main():
    # 文件路径
    # aria_file = "/home/pine/RoboTwin2/sfu_cooking_008_5/aria01.json"
    # wrist_file = "/home/pine/RoboTwin2/sfu_cooking_008_5/wrist_data.json"
    # combined_file = "/Users/haoyuan/Desktop/ICRA2025/bimanual_cross_painting/0731newdata/0_wrist_data.json" 
    combined_file = "/data1/zjyang/program/egodex/egodex_stored/clean_surface/0_wrist_data.json" 
    
    print("加载数据文件...")
    
    # 加载数据
    # aria_data = load_json_file(aria_file)
    # wrist_data = load_json_file(wrist_file)
    combined_data = load_json_file(combined_file)

    # if aria_data is None or wrist_data is None:
    #     print("无法加载数据文件")
    #     return
    
    # 提取位置数据
    print("提取位置数据...")
    camera_positions = extract_camera_positions_combined(combined_data)
    camera_orientations = extract_camera_orientations(combined_data)
    left_wrist_positions, right_wrist_positions = extract_wrist_positions_combined(combined_data, max_points=5264)
    left_wrist_orientations, right_wrist_orientations = extract_wrist_orientation_combined(combined_data, max_points=5264)

    # 由于时间戳不完全对齐，我们使用较小的数据集长度
    max_frames = 30
    
    min_length = min(len(camera_positions), len(left_wrist_positions), len(right_wrist_positions), max_frames)
    if min_length > 0:
        camera_positions = camera_positions[:min_length]
        camera_orientations = camera_orientations[:min_length]
        left_wrist_positions = left_wrist_positions[:min_length]
        left_wrist_orientations = left_wrist_orientations[:min_length]
        right_wrist_positions = right_wrist_positions[:min_length]
        right_wrist_orientations = right_wrist_orientations[:min_length]
        print(f"使用前{min_length}帧数据进行可视化")
        
    # 绘制2D图
    print("生成2D俯视图...")
    plot_2d_trajectories(camera_positions, left_wrist_positions, right_wrist_positions)
    
    # 保存图片
    plt.savefig('camera_wrist_trajectories_2d.png', dpi=300, bbox_inches='tight')
    print("图片已保存为: camera_wrist_trajectories_2d.png")
    
    # 显示图片
    plt.show()
    
    # 新的 3D 图绘制
    # print("生成3D轨迹图...")
    # plot_3d_trajectories(camera_positions, left_wrist_positions, right_wrist_positions)
    # plt.savefig('camera_wrist_trajectories_3d.png', dpi=300, bbox_inches='tight')
    # print("图片已保存为: camera_wrist_trajectories_3d.png")
    print("生成3D轨迹图并绘制朝向...")
    plot_3d_trajectories(camera_positions, left_wrist_positions,left_wrist_orientations, right_wrist_positions, right_wrist_orientations, camera_orientations)
    plt.savefig('camera_wrist_trajectories_3d.png', dpi=300, bbox_inches='tight')
    print("图片已保存为: camera_wrist_trajectories_3d.png")

    # print_statistics(camera_positions, left_wrist_positions, right_wrist_positions)
    
    # # 创建可视化几何体
    # print("创建3D可视化...")
    # geometries = []
    
    # # 添加坐标系
    # geometries.append(add_coordinate_frame())
    
    # # 添加相机轨迹（黑色）
    # camera_geometries = create_camera_visualization(camera_positions, color=[0, 0, 0])
    # geometries.extend(camera_geometries)
    
    # # 添加左手腕轨迹（蓝色）
    # left_wrist_geometries = create_wrist_visualization(left_wrist_positions, color=[0, 0, 1], name="Left Wrist")
    # geometries.extend(left_wrist_geometries)
    
    # # 添加右手腕轨迹（绿色）
    # right_wrist_geometries = create_wrist_visualization(right_wrist_positions, color=[0, 1, 0], name="Right Wrist")
    # geometries.extend(right_wrist_geometries)
    
    # # 显示可视化
    # print("\n开启3D可视化窗口...")
    # print("颜色说明:")
    # print("  - 黑色: 相机轨迹")
    # print("  - 蓝色: 左手腕轨迹")
    # print("  - 绿色: 右手腕轨迹")
    # print("  - 浅色球: 轨迹起始点")
    # print("  - 深色球: 轨迹结束点")
    # print("\n操作提示:")
    # print("  - 鼠标左键拖拽: 旋转视角")
    # print("  - 鼠标右键拖拽: 平移视角")
    # print("  - 滚轮: 缩放")
    # print("  - 按 'Q' 或关闭窗口退出")
    
    # # 创建可视化窗口
    # vis = o3d.visualization.Visualizer()
    # vis.create_window(window_name="相机和手腕轨迹可视化", width=1200, height=800)
    
    # # 添加所有几何体
    # for geom in geometries:
    #     vis.add_geometry(geom)
    
    # # 设置渲染选项
    # render_option = vis.get_render_option()
    # render_option.background_color = np.array([0.95, 0.95, 0.95])  # 浅灰色背景
    # render_option.point_size = 3.0
    # render_option.line_width = 2.0
    
    # # 运行可视化
    # vis.run()
    # vis.destroy_window()
    
    # print("可视化完成!")

if __name__ == "__main__":
    # 设置中文字体支持
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    main()


import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from mpl_toolkits.mplot3d import Axes3D  # 导入3D绘图工具
import os
import json
from tqdm import tqdm  # 添加进度条支持
import collections  # 用于轨迹历史记录

# 导入必要的函数
from extrinsic_wrist_traj_viewer import (
    load_json_file,
    extract_camera_positions_combined,
    extract_camera_orientations,
    extract_wrist_positions_combined,
    extract_wrist_orientation_combined
)

def transform_to_camera_frame(point, camera_position, camera_orientation):
    """
    将世界坐标系中的点转换到相机坐标系
    
    Parameters:
    - point: 世界坐标系中的点 [x, y, z]
    - camera_position: 相机在世界坐标系中的位置 [x, y, z]
    - camera_orientation: 相机旋转矩阵 (3x3)
    
    Returns:
    - 相机坐标系中的点 [x, y, z]
    """
    # 确保输入是numpy数组
    point = np.array(point)
    camera_position = np.array(camera_position)
    camera_orientation = np.array(camera_orientation)
    
    # 将旋转矩阵从形状(9,)转换为(3,3)
    if camera_orientation.shape == (9,):
        camera_orientation = camera_orientation.reshape(3, 3)
    
    # 计算点相对于相机的位置向量
    relative_position = point - camera_position
    
    # 使用相机旋转矩阵的转置将点从世界坐标系转换到相机坐标系
    # 注意：旋转矩阵的转置等于其逆矩阵（对于正交矩阵）
    camera_frame_point = np.dot(camera_orientation.T, relative_position)
    
    return camera_frame_point

def draw_wrist_positions_on_frame(frame, left_wrist_position, right_wrist_position, camera_position, camera_orientation, camera_matrix=None, 
                             wrist_trail_history=None, trail_length=30):
    """
    Directly annotate the real-time position points and trajectories of the left and right wrists on the video frame
    
    Parameters:
    - frame: video frame
    - left_wrist_position: left wrist position in world coordinates
    - right_wrist_position: right wrist position in world coordinates
    - camera_position: camera position in world coordinates
    - camera_orientation: camera orientation (rotation matrix)
    - camera_matrix: camera intrinsic matrix
    - wrist_trail_history: wrist trajectory history
    - trail_length: trajectory length
    """
    h, w = frame.shape[:2]
    
    # 如果没有提供轨迹历史记录，则创建一个空的
    if wrist_trail_history is None:
        wrist_trail_history = {
            'left': collections.deque(maxlen=trail_length),
            'right': collections.deque(maxlen=trail_length)
        }
    
    # 将世界坐标系中的手腕位置转换到相机坐标系
    left_wrist_camera = transform_to_camera_frame(left_wrist_position, camera_position, camera_orientation)
    right_wrist_camera = transform_to_camera_frame(right_wrist_position, camera_position, camera_orientation)
    
    # 将3D点投影到2D图像平面
    if camera_matrix is not None:
        # 使用相机内参进行投影
        fx = camera_matrix[0][0]
        fy = camera_matrix[1][1]
        cx = camera_matrix[0][2]
        cy = camera_matrix[1][2]
        
        # 左手 wrist 相机坐标系下的3D坐标
        Xl, Yl, Zl = left_wrist_camera
        
        # 确保Z值为正，否则点在相机后方，不可见
        if Zl > 0:
            ul = int(fx * Xl / Zl + cx)
            vl = int(fy * Yl / Zl + cy)
            left_pos_2d = (ul, vl)
        else:
            # 如果点在相机后方，使用屏幕外的坐标
            left_pos_2d = (-100, -100)

        # 右手 wrist 相机坐标系下的3D坐标
        Xr, Yr, Zr = right_wrist_camera
        
        if Zr > 0:
            ur = int(fx * Xr / Zr + cx)
            vr = int(fy * Yr / Zr + cy)
            right_pos_2d = (ur, vr)
        else:
            right_pos_2d = (-100, -100)
            
        # 打印调试信息
        print(f"相机坐标系下 - 左手: {left_wrist_camera}, 右手: {right_wrist_camera}")
        print(f"投影到图像 - 左手: {left_pos_2d}, 右手: {right_pos_2d}")

    else:
        # 如果没有相机内参，使用简单的缩放
        left_pos_2d = (int(left_wrist_camera[0] * w/2 + w/2), int(left_wrist_camera[1] * h/2 + h/2))
        right_pos_2d = (int(right_wrist_camera[0] * w/2 + w/2), int(right_wrist_camera[1] * h/2 + h/2))
    
    # 只有当点在视野内时才添加到轨迹历史
    if 0 <= left_pos_2d[0] < w and 0 <= left_pos_2d[1] < h:
        wrist_trail_history['left'].append(left_pos_2d)
    
    if 0 <= right_pos_2d[0] < w and 0 <= right_pos_2d[1] < h:
        wrist_trail_history['right'].append(right_pos_2d)
    
    # 绘制左手腕轨迹
    for i in range(1, len(wrist_trail_history['left'])):
        # 轨迹颜色随着时间渐变（越近的点越亮）
        alpha = i / len(wrist_trail_history['left'])
        color = (0, 0, int(139 + 116 * alpha))  # 从深红色到亮红色
        thickness = max(1, int(3 * alpha))  # 越近的点越粗
        
        pt1 = wrist_trail_history['left'][i-1]
        pt2 = wrist_trail_history['left'][i]
        cv2.line(frame, pt1, pt2, color, thickness)
    
    # 绘制右手腕轨迹
    for i in range(1, len(wrist_trail_history['right'])):
        alpha = i / len(wrist_trail_history['right'])
        color = (int(69 + 186 * alpha), int(69 * alpha), 255)  # 从橙红色到亮蓝色
        thickness = max(1, int(3 * alpha))
        
        pt1 = wrist_trail_history['right'][i-1]
        pt2 = wrist_trail_history['right'][i]
        cv2.line(frame, pt1, pt2, color, thickness)
    
    # 绘制当前左手腕位置
    cv2.circle(frame, left_pos_2d, 6, (0, 0, 255), -1)  # 红色实心圆，稍微小一点
    cv2.circle(frame, left_pos_2d, 8, (255, 255, 255), 2)  # 白色边框
    cv2.putText(frame, "L", (left_pos_2d[0] - 10, left_pos_2d[1] - 10),  # 简化标签
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    # 绘制当前右手腕位置
    cv2.circle(frame, right_pos_2d, 6, (255, 0, 0), -1)  # 蓝色实心圆，稍微小一点
    cv2.circle(frame, right_pos_2d, 8, (255, 255, 255), 2)  # 白色边框
    cv2.putText(frame, "R", (right_pos_2d[0] - 10, right_pos_2d[1] - 10),  # 简化标签
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    
    return frame, wrist_trail_history

# 轨迹窗口功能已移除

def draw_pose_on_image(frame, origin, orientation, colors, labels, camera_position=None, camera_orientation=None, camera_matrix=None, scale=100, position_override=None, fixed_arrow_length=20):
    """
    Draw pose arrows on the image
    
    Parameters:
    - frame: video frame
    - origin: origin position in world coordinates
    - orientation: orientation matrix in world coordinates
    - colors: colors for each axis
    - labels: labels for each axis
    - camera_position: camera position in world coordinates
    - camera_orientation: camera orientation matrix
    - camera_matrix: camera intrinsic matrix
    - scale: scale factor for arrows
    - position_override: optional tuple (x, y) to override the position of the origin in the image
    - fixed_arrow_length: fixed length of arrows in pixels
    """
    # 将3D点投影到2D图像平面
    h, w = frame.shape[:2]
    
    # 如果提供了位置覆盖，直接使用它
    if position_override is not None:
        origin_2d = position_override
    else:
        # 如果提供了相机位置和方向，将世界坐标系中的点转换到相机坐标系
        if camera_position is not None and camera_orientation is not None:
            # 转换原点到相机坐标系
            origin_camera = transform_to_camera_frame(origin, camera_position, camera_orientation)
            
            if camera_matrix is not None:
                # 使用相机内参进行投影
                fx = camera_matrix[0][0]
                fy = camera_matrix[1][1]
                cx = camera_matrix[0][2]
                cy = camera_matrix[1][2]
                
                # 只有当Z值为正时才进行投影（点在相机前方）
                if origin_camera[2] > 0:
                    origin_2d = (int(fx * origin_camera[0] / origin_camera[2] + cx), 
                                int(fy * origin_camera[1] / origin_camera[2] + cy))
                else:
                    # 点在相机后方，设置为屏幕外
                    origin_2d = (-100, -100)
            else:
                # 如果没有相机内参，使用简单的缩放
                origin_2d = (int(origin_camera[0] * w/2 + w/2), int(origin_camera[1] * h/2 + h/2))
        elif camera_matrix is not None:
            # 如果只有相机内参但没有位置和方向（兼容旧代码）
            fx = camera_matrix[0][0]
            fy = camera_matrix[1][1]
            cx = camera_matrix[0][2]
            cy = camera_matrix[1][2]
            origin_2d = (int(origin[0] * fx + cx), int(origin[1] * fy + cy))
        else:
            # 如果没有相机内参，使用简单的缩放（保持原来的逻辑作为备选）
            origin_2d = (int(origin[0] * w/2 + w/2), int(origin[1] * h/2 + h/2))
    
    # 检查原点是否在视野内
    if origin_2d[0] < 0 or origin_2d[0] >= w or origin_2d[1] < 0 or origin_2d[1] >= h:
        # 原点不在视野内，不绘制任何东西
        return
    
    # 确保方向矩阵是正确的形状
    orientation_np = np.array(orientation)
    if orientation_np.shape == (9,):
        orientation_np = orientation_np.reshape(3, 3)
    elif orientation_np.shape != (3, 3):
        print(f"警告：朝向矩阵形状不正确: {orientation_np.shape}，应为 (3, 3) 或 (9,)")
        # 如果形状不正确，创建一个单位矩阵作为替代
        orientation_np = np.eye(3)
    
    # 打印调试信息
    if "Right" in labels[0] or "R-" in labels[0]:
        print(f"右手朝向矩阵: {orientation_np}")
    
    for i, (color, label) in enumerate(zip(colors, labels)):
        # 获取方向向量
        try:
            direction = orientation_np[:, i]
        except IndexError:
            print(f"错误：无法获取朝向矩阵的第 {i} 列，矩阵形状: {orientation_np.shape}")
            direction = np.array([1.0 if i == 0 else 0.0, 1.0 if i == 1 else 0.0, 1.0 if i == 2 else 0.0])
        
        # 获取方向向量的单位向量
        direction_norm = np.linalg.norm(direction)
        if direction_norm > 0:
            unit_direction = direction / direction_norm
        else:
            unit_direction = direction
            
        # 如果提供了相机位置和方向，需要转换方向向量
        if camera_position is not None and camera_orientation is not None and position_override is None:
            # 计算方向向量的终点（在世界坐标系中）
            # 使用固定长度的箭头
            end_point_world = origin + unit_direction * 0.05  # 使用小的世界坐标单位长度
            
            # 将终点转换到相机坐标系
            end_point_camera = transform_to_camera_frame(end_point_world, camera_position, camera_orientation)
            
            if camera_matrix is not None and end_point_camera[2] > 0:
                # 使用相机内参投影终点
                fx = camera_matrix[0][0]
                fy = camera_matrix[1][1]
                cx = camera_matrix[0][2]
                cy = camera_matrix[1][2]
                
                end_point = (int(fx * end_point_camera[0] / end_point_camera[2] + cx),
                            int(fy * end_point_camera[1] / end_point_camera[2] + cy))
            else:
                # 如果终点在相机后方或没有相机内参，不绘制这个箭头
                continue
        else:
            # 使用固定长度的箭头
            end_point = (
                int(origin_2d[0] + unit_direction[0] * fixed_arrow_length),
                int(origin_2d[1] + unit_direction[1] * fixed_arrow_length)
            )
        
        # 将颜色名称转换为BGR值
        if isinstance(color, str):
            color_map = {
                'red': (0, 0, 255),
                'green': (0, 255, 0),
                'blue': (255, 0, 0),
                'darkred': (0, 0, 139),
                'darkgreen': (0, 139, 0),
                'darkblue': (139, 0, 0),
                'orangered': (0, 69, 255),
                'lime': (0, 255, 0),
                'skyblue': (235, 206, 135)
            }
            bgr_color = color_map.get(color, (0, 0, 255))
        else:
            bgr_color = color
            
        # 绘制箭头，使用更细的线条和更小的箭头头部
        cv2.arrowedLine(frame, origin_2d, end_point, bgr_color, 1, tipLength=0.3)
        
        # 添加标签（更小的字体和更靠近箭头）
        cv2.putText(frame, label, (end_point[0] + 2, end_point[1] + 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, bgr_color, 1)

# 3D坐标窗口功能已移除

def visualize_poses_in_video(video_path, output_path, camera_positions, left_wrist_positions, right_wrist_positions, 
                           camera_orientations=None, left_wrist_orientations=None, right_wrist_orientations=None, 
                           camera_matrix=None, skip=1, scale=0.1, 
                           show_wrist_positions=True, wrist_trail_length=30):
    """
    Visualize pose information of wrists and embed it into the video.
    
    Parameters:
    - video_path: path to the input video
    - output_path: path to save the output video
    - camera_positions: camera positions in world coordinates
    - left_wrist_positions: left wrist positions in world coordinates
    - right_wrist_positions: right wrist positions in world coordinates
    - camera_orientations: camera orientations (rotation matrices)
    - left_wrist_orientations: left wrist orientations (rotation matrices)
    - right_wrist_orientations: right wrist orientations (rotation matrices)
    - camera_matrix: camera intrinsic matrix
    - skip: frame skip for pose visualization
    - scale: scale factor for arrows
    - show_wrist_positions: whether to display wrist positions
    - wrist_trail_length: wrist trail history length
    """
    # 检查输入文件是否存在
    if not os.path.exists(video_path):
        print(f"Error: Video file does not exist: {video_path}")
        return

    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Cannot open video file")
        return

    # 获取视频信息
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 创建输出目录
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    # 创建视频写入对象
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # 确保数据长度一致
    min_length = min(len(camera_positions), len(left_wrist_positions), len(right_wrist_positions))
    if min_length == 0:
        print("Error: No valid pose data")
        return

    # 设置颜色和标签
    camera_colors = ['red', 'green', 'blue']
    left_wrist_colors = ['darkred', 'darkgreen', 'darkblue']
    right_wrist_colors = ['orangered', 'lime', 'skyblue']
    axis_labels = ['X', 'Y', 'Z']
    
    # 轨迹窗口功能已移除
    
    # 初始化手腕轨迹历史记录，用于直接在视频上标注手腕位置
    wrist_trail_history = None if not show_wrist_positions else {
        'left': collections.deque(maxlen=wrist_trail_length),
        'right': collections.deque(maxlen=wrist_trail_length)
    }

    # 遍历视频帧
    frame_idx = 0
    pbar = tqdm(total=min(total_frames, min_length), desc="Processing video frames")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame_idx >= min_length:
            break

        # 在原始帧上绘制位姿
        img = frame.copy()
        
        # 轨迹记录功能已移除
            
        # 在视频上直接标注手腕位置和轨迹
        if show_wrist_positions:
            img, wrist_trail_history = draw_wrist_positions_on_frame(
                img, 
                left_wrist_positions[frame_idx],
                right_wrist_positions[frame_idx],
                camera_positions[frame_idx],
                camera_orientations[frame_idx] if camera_orientations is not None else None,
                camera_matrix=camera_matrix,
                wrist_trail_history=wrist_trail_history,
                trail_length=wrist_trail_length
            )
            
            # 打印调试信息，检查右手朝向数据
            if frame_idx % 30 == 0 and right_wrist_orientations is not None:
                print(f"Frame {frame_idx}:")
                print(f"左手朝向: {left_wrist_orientations[frame_idx]}")
                print(f"右手朝向: {right_wrist_orientations[frame_idx]}")
                print("---")
        
        if frame_idx % skip == 0:
            # 相机位姿显示已移除

            # 绘制左手腕位姿
            if left_wrist_orientations is not None:
                draw_pose_on_image(img, left_wrist_positions[frame_idx],
                                left_wrist_orientations[frame_idx],
                                left_wrist_colors,
                                [f'L-{axis}' for axis in axis_labels],  # 简化标签
                                camera_position=camera_positions[frame_idx],
                                camera_orientation=camera_orientations[frame_idx] if camera_orientations is not None else None,
                                camera_matrix=camera_matrix,
                                scale=scale * 0.5,  # 减小箭头大小
                                fixed_arrow_length=15)  # 使用固定长度的箭头

            # 绘制右手腕位姿
            if right_wrist_orientations is not None:
                draw_pose_on_image(img, right_wrist_positions[frame_idx],
                                right_wrist_orientations[frame_idx],
                                right_wrist_colors,
                                [f'R-{axis}' for axis in axis_labels],  # 简化标签
                                camera_position=camera_positions[frame_idx],
                                camera_orientation=camera_orientations[frame_idx] if camera_orientations is not None else None,
                                camera_matrix=camera_matrix,
                                scale=scale * 0.5,  # 减小箭头大小
                                fixed_arrow_length=15)  # 使用固定长度的箭头

            # 添加帧号
            cv2.putText(img, f'Frame: {frame_idx}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # 轨迹窗口已移除
            
            # 3D坐标窗口已移除

        # 写入输出视频
        out.write(img)
        
        frame_idx += 1
        pbar.update(1)

    # 清理资源
    pbar.close()
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Video saved as: {output_path}")

if __name__ == "__main__":
    # 设置任务名称和路径
    # task_name = "clean_cups" #"clean_surface"
    base_dir = "/home/dhy/RoboTwin2"
    output_dir = "traj_test2"
    os.makedirs(output_dir, exist_ok=True)  # 确保输出目录存在
    # task_name = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    task_name =["clean_cups"]
    for task in task_name:
        video_path = f"{base_dir}/{task}/0.mp4"
        output_path = f"{output_dir}/output_video_with_poses_{task}.mp4"
        
        # 检查视频文件是否存在
        if not os.path.exists(video_path):
            print(f"Warning: Video file does not exist: {video_path}")
            continue
    
        try:
            # 加载位姿数据
            combined_file = f"/home/dhy/RoboTwin2/{task}/0_wrist_data.json"
            combined_data = load_json_file(combined_file)
            
            if combined_data is None:
                raise ValueError(f"Cannot load data file: {combined_file}")

            # 从JSON中获取相机内参，如果没有则使用默认值
            camera_intrinsic = None # ombined_data.get('camera', {}).get('intrinsic')
            
            # 如果JSON中没有相机内参，使用默认值
            if camera_intrinsic is None:
                # 使用提供的默认内参矩阵
                camera_intrinsic = [
                    [736.6339111328125, 0.0, 960.0],
                    [0.0, 736.6339111328125, 540.0],
                    [0.0, 0.0, 1.0]
                ]
                print("Using default camera intrinsic matrix")
            
            # 提取位姿数据
            camera_positions = extract_camera_positions_combined(combined_data)
            camera_orientations = extract_camera_orientations(combined_data)
            left_wrist_positions, right_wrist_positions = extract_wrist_positions_combined(combined_data)
            left_wrist_orientations, right_wrist_orientations = extract_wrist_orientation_combined(combined_data)

            # 调用可视化函数
            visualize_poses_in_video(
                video_path=video_path,
                output_path=output_path,
                camera_positions=camera_positions,
                left_wrist_positions=left_wrist_positions,
                right_wrist_positions=right_wrist_positions,
                camera_orientations=camera_orientations,
                left_wrist_orientations=left_wrist_orientations,
                right_wrist_orientations=right_wrist_orientations,
                camera_matrix=camera_intrinsic,  # 添加相机内参
                skip=1,  # 每帧都绘制
                scale=1,  # 缩小scale值，使坐标轴更小
                show_wrist_positions=True,  # 在视频上直接标注手腕位置
                wrist_trail_length=30  # 手腕轨迹长度
            )
        except Exception as e:
            print(f"Error: {str(e)}")    