# data/calibration.py

import numpy as np

class Calibration:
    """标定数据处理类"""

    def __init__(self, cfg=None, calib_file=None):
        self.calib_data = {}
        if calib_file:
            self.load_calibration(calib_file)
        elif cfg:
            # 从配置初始化
            self.init_from_config(cfg)

    def init_from_config(self, cfg):
        """从配置初始化标定参数"""
        calib_cfg = cfg.get('calibration', {})

        # 设置默认标定参数（使用题目给出的矩阵）
        self.calib_data = {
            'P0': np.array(calib_cfg.get('P0', [
                [605.6403, 0.0, 319.2964, 0.0],
                [0.0, 605.6746, 235.4414, 0.0],
                [0.0, 0.0, 1.0, 0.0]
            ])),
            'Tr_velo_to_cam': np.array(calib_cfg.get('Tr_velo_to_cam', [
                [0, -1, 0, -0.25],
                [0, 0, -1, 0.4],
                [1, 0, 0, -0.25],
                [0, 0, 0, 1]
            ])),
            'R0_rect': np.array(calib_cfg.get('R0_rect', [
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1]
            ]))
        }

    def load_calibration(self, calib_path):
        """加载KITTI格式标定文件"""
        calib_data = {}
        try:
            with open(calib_path, 'r') as f:
                for line in f.readlines():
                    if ':' in line:
                        key, value = line.split(':', 1)
                        key = key.strip()
                        values = [float(x) for x in value.strip().split()]
                        
                        # 根据KITTI格式处理不同参数
                        if key == 'P0':  # 相机0的内参
                            calib_data['P0'] = np.array(values).reshape(3, 4)
                        elif key == 'P1':  # 相机1的内参
                            calib_data['P1'] = np.array(values).reshape(3, 4)
                        elif key == 'P2':  # 相机2的内参
                            calib_data['P2'] = np.array(values).reshape(3, 4)
                        elif key == 'P3':  # 相机3的内参
                            calib_data['P3'] = np.array(values).reshape(3, 4)
                        elif key == 'R0_rect':  # 旋转修正矩阵
                            calib_data['R0_rect'] = np.array(values).reshape(3, 3)
                        elif key == 'Tr_velo_to_cam':  # 雷达到相机的变换
                            calib_data['Tr_velo_to_cam'] = np.array(values).reshape(3, 4)
                        elif key == 'Tr_imu_to_velo':  # IMU到雷达的变换
                            calib_data['Tr_imu_to_velo'] = np.array(values).reshape(3, 4)
            
            self.calib_data = calib_data
            return True
            
        except Exception as e:
            print(f"Error loading calibration file {calib_path}: {e}")
            return False
    
    def get_camera_matrix(self, camera_id=0):
        """获取相机内参矩阵"""
        key = f'P{camera_id}'
        return self.calib_data.get(key, None)
    
    def get_velo_to_cam_matrix(self):
        """获取雷达到相机的变换矩阵"""
        return self.calib_data.get('Tr_velo_to_cam', None)
    
    def get_rectification_matrix(self):
        """获取旋转修正矩阵"""
        return self.calib_data.get('R0_rect', np.eye(3))

    def project_velo_to_image(self, points_3d, camera_id=0):
        # 获取变换矩阵
        R_rect = self.get_rectification_matrix()
        T_velo_to_cam = self.get_velo_to_cam_matrix()
        P_cam = self.get_camera_matrix(camera_id)

        if T_velo_to_cam is None or P_cam is None:
            return None

        # 扩展点为齐次坐标 (N, 4)
        points_homo = np.hstack([points_3d[:, :3], np.ones((points_3d.shape[0], 1))])

        # 确保变换矩阵是4x4
        if T_velo_to_cam.shape == (3, 4):
            T_velo_to_cam_homo = np.vstack([T_velo_to_cam, [0, 0, 0, 1]])
        else:
            T_velo_to_cam_homo = T_velo_to_cam

        # 矩阵扩展为 4x4
        R_rect_homo = np.eye(4)
        R_rect_homo[:3, :3] = R_rect

        # 矩阵变换：雷达坐标系 -> 相机坐标系 -> 修正后相机坐标系
        points_cam = np.dot(T_velo_to_cam_homo, points_homo.T).T  # (N, 4)
        points_rect = np.dot(R_rect_homo, points_cam.T).T  # (N, 4)

        # 提取 3D 坐标并扩展为齐次坐标（用于投影）
        points_rect_3d = points_rect[:, :3]  # (N, 3)
        points_rect_homo = np.hstack([points_rect_3d, np.ones((points_rect_3d.shape[0], 1))])  # (N, 4)

        # 投影到图像平面：相机内参 (3,4) × 齐次坐标 (4,N) -> (3,N)
        points_image = np.dot(P_cam, points_rect_homo.T).T  # (N, 3)

        # 归一化（除以 z 分量）
        points_image[:, 0] /= points_image[:, 2]
        points_image[:, 1] /= points_image[:, 2]

        return points_image[:, :2]  # 返回 (u, v) 坐标

    def project_image_to_velo(self, points_2d, depths, camera_id=0):
        P_cam = self.get_camera_matrix(camera_id)
        R_rect = self.get_rectification_matrix()
        T_velo_to_cam = self.get_velo_to_cam_matrix()

        if P_cam is None or T_velo_to_cam is None:
            return None

        # 构造齐次变换矩阵
        T_velo_to_cam_homo = np.vstack([T_velo_to_cam, [0, 0, 0, 1]])
        T_cam_to_velo_homo = np.linalg.inv(T_velo_to_cam_homo)  # 稳定的逆矩阵计算

        # 图像点 -> 相机坐标系
        points_img_homo = np.hstack([points_2d, np.ones((len(points_2d), 1))])  # [N, 3]
        points_cam = np.dot(points_img_homo, np.linalg.inv(P_cam[:, :3]).T)  # [N, 3]
        points_cam *= depths.reshape(-1, 1)  # 缩放至实际深度

        # 修正坐标系变换
        points_rect = np.dot(R_rect, points_cam.T).T  # [N, 3]
        points_rect_homo = np.hstack([points_rect, np.ones((len(points_rect), 1))])  # [N, 4]

        # 相机坐标系 -> 雷达坐标系
        points_velo = np.dot(points_rect_homo, T_cam_to_velo_homo.T)  # [N, 4]
        return points_velo[:, :3]  # 返回3D坐标

    def get_focal_length(self, camera_id=0):
        """获取相机焦距"""
        P = self.get_camera_matrix(camera_id)
        if P is not None:
            return P[0, 0], P[1, 1]  # fx, fy
        return None, None

    def get_principal_point(self, camera_id=0):
        """获取主点坐标"""
        P = self.get_camera_matrix(camera_id)
        if P is not None:
            return P[0, 2], P[1, 2]  # cx, cy
        return None, None