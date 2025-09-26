# scripts/inference.py
import torch
import os
import yaml
from torch.utils.data import DataLoader
import numpy as np

from data.dataset import DRadDataset
from models.fusion_model import RadarCameraFusionModel
from utils.metrics import DetectionMetrics


def convert_predictions_to_boxes(predictions, batch_data, data_cfg):
    """将模型预测转换为边界框格式"""
    boxes = []

    batch_size = predictions['batch_size']
    pc_cfg = data_cfg['pointcloud']
    x_range = pc_cfg['x_range']
    y_range = pc_cfg['y_range']
    grid_h, grid_w = 32, 32

    for i in range(batch_size):
        # 解析预测结果
        cls_pred = predictions['cls_pred'][i]  # [C, H, W]
        reg_pred = predictions['reg_pred'][i]  # [7, H, W]

        # 使用阈值过滤
        cls_probs = torch.sigmoid(cls_pred)
        max_probs, max_indices = torch.max(cls_probs, dim=0)

        height, width = cls_probs.shape[1:]
        cell_size_x = (x_range[1] - x_range[0]) / grid_w
        cell_size_y = (y_range[1] - y_range[0]) / grid_h

        # 遍历BEV网格生成边界框
        for h in range(height):
            for w in range(width):
                if max_probs[h, w] > 0.3:  # 置信度阈值
                    class_id = max_indices[h, w].item()
                    reg_data = reg_pred[:, h, w]

                    class_names = ['Car', 'Cyclist', 'Truck']
                    if class_id >= len(class_names):
                        continue

                    # 将相对坐标转换为绝对坐标
                    abs_x = w * cell_size_x + x_range[0] + reg_data[0] * cell_size_x
                    abs_y = h * cell_size_y + y_range[0] + reg_data[1] * cell_size_y
                    abs_z = reg_data[2]

                    box = {
                        'class': class_names[class_id],
                        'bbox_3d': [
                            abs_x.item(),  # x
                            abs_y.item(),  # y
                            abs_z.item(),  # z
                            reg_data[5].item(),  # height
                            reg_data[3].item(),  # width
                            reg_data[4].item(),  # length
                            reg_data[6].item()  # rotation
                        ],
                        'score': max_probs[h, w].item()
                    }
                    boxes.append(box)

    return boxes


def convert_labels_to_boxes(batch_data):
    """将标签数据转换为边界框格式"""
    boxes = []

    batch_size = len(batch_data['labels'])

    for i in range(batch_size):
        if batch_data['labels'][i] is None:
            continue

        for label in batch_data['labels'][i]:
            box = {
                'class': label['type'],
                'bbox_3d': [
                    label['location'][0],  # x
                    label['location'][1],  # y
                    label['location'][2],  # z
                    label['dimensions'][0],  # height
                    label['dimensions'][1],  # width
                    label['dimensions'][2],  # length
                    label['rotation_y']  # rotation
                ]
            }
            boxes.append(box)

    return boxes


def collate_fn(batch):
    """自定义批次整理函数"""
    if len(batch) == 0:
        return {}

    max_voxels = max(len(item['voxels']) for item in batch)
    feature_dim = batch[0]['voxels'].shape[-1] if len(batch[0]['voxels']) > 0 else 8

    padded_voxels = []
    padded_coords = []
    padded_numpoints = []

    for item in batch:
        voxels = item['voxels']
        coords = item['coordinates']
        numpoints = item['num_points']

        if len(voxels) < max_voxels:
            pad_size = max_voxels - len(voxels)
            padded_voxels.append(torch.cat([
                voxels,
                torch.zeros(pad_size, voxels.shape[1], feature_dim)
            ], dim=0))
            padded_coords.append(torch.cat([
                coords,
                torch.zeros(pad_size, 3, dtype=coords.dtype)
            ], dim=0))
            padded_numpoints.append(torch.cat([
                numpoints,
                torch.zeros(pad_size, dtype=numpoints.dtype)
            ], dim=0))
        else:
            padded_voxels.append(voxels[:max_voxels])
            padded_coords.append(coords[:max_voxels])
            padded_numpoints.append(numpoints[:max_voxels])

    return {
        'voxels': torch.stack(padded_voxels),
        'coordinates': torch.stack(padded_coords),
        'num_points': torch.stack(padded_numpoints),
        'image': torch.stack([item['image'] for item in batch]),
        'calib': [item['calib'] for item in batch],
        'sample_id': [item['sample_id'] for item in batch],
        'labels': [item.get('labels', []) for item in batch]
    }


def move_to_device(data, device):
    """将数据移动到指定设备"""
    if isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, dict):
        return {k: move_to_device(v, device) for k, v in data.items()}
    elif isinstance(data, list):
        return [move_to_device(item, device) for item in data]
    else:
        return data


def calculate_truncation(bbox_2d, image_size=(640, 480)):
    """计算2D边界框的截断程度"""
    if bbox_2d is None or len(bbox_2d) != 4:
        return 0.0

    img_width, img_height = image_size
    x_min, y_min, x_max, y_max = bbox_2d

    # 计算边界框面积
    bbox_area = (x_max - x_min) * (y_max - y_min)
    if bbox_area <= 0:
        return 0.0

    # 计算图像内可见区域
    visible_x_min = max(0, x_min)
    visible_y_min = max(0, y_min)
    visible_x_max = min(img_width, x_max)
    visible_y_max = min(img_height, y_max)

    # 计算可见区域面积
    visible_area = max(0, visible_x_max - visible_x_min) * max(0, visible_y_max - visible_y_min)

    # 截断程度 = 1 - 可见比例
    truncation = 1.0 - (visible_area / bbox_area)

    return max(0.0, min(1.0, truncation))


def estimate_occlusion(box_3d):
    """估计3D边界框的遮挡程度（简化版本）"""
    # 这是一个简化实现，实际应用中需要更复杂的逻辑
    # 基于边界框的位置和尺寸进行简单估计

    location = box_3d['location']
    dimensions = box_3d['dimensions']

    # 距离相机越远，遮挡可能性越大
    distance = np.sqrt(location[0] ** 2 + location[1] ** 2 + location[2] ** 2)

    # 物体尺寸越小，遮挡可能性越大
    volume = dimensions[0] * dimensions[1] * dimensions[2]

    # 简单启发式规则
    if distance > 50:  # 距离超过50米
        return 2  # 严重遮挡
    elif distance > 25:  # 距离25-50米
        return 1  # 部分遮挡
    else:
        return 0  # 完全可见


def calculate_alpha(box_3d, calib=None):
    """计算观测角度alpha"""
    if calib is None:
        return 0.0

    try:
        location = box_3d['location']
        rotation = box_3d.get('rotation', 0)

        # 将3D点从雷达坐标系转换到相机坐标系
        if len(location) >= 3:
            point_3d = np.array([location[0], location[1], location[2], 1.0])

            # 获取变换矩阵
            T_velo_to_cam = calib.get_velo_to_cam_matrix()
            R_rect = calib.get_rectification_matrix()

            if T_velo_to_cam is not None and R_rect is not None:
                # 扩展为4x4矩阵
                if T_velo_to_cam.shape == (3, 4):
                    T_velo_to_cam_4x4 = np.eye(4)
                    T_velo_to_cam_4x4[:3, :] = T_velo_to_cam
                else:
                    T_velo_to_cam_4x4 = T_velo_to_cam

                # 应用变换
                point_cam = np.dot(T_velo_to_cam_4x4, point_3d)

                # 计算观测角度 alpha = rotation_y - theta
                theta = np.arctan2(point_cam[0], point_cam[2])
                alpha = rotation - theta

                # 将角度归一化到 [-pi, pi]
                while alpha > np.pi:
                    alpha -= 2 * np.pi
                while alpha < -np.pi:
                    alpha += 2 * np.pi

                return alpha

    except Exception as e:
        print(f"计算alpha角度时出错: {e}")

    return 0.0


def compute_3d_box_corners(x, y, z, w, l, h, rotation):
    """计算3D边界框的8个角点"""
    # 创建相对于中心的角点
    corners = np.array([
        [l / 2, w / 2, h / 2], [l / 2, w / 2, -h / 2], [l / 2, -w / 2, h / 2], [l / 2, -w / 2, -h / 2],
        [-l / 2, w / 2, h / 2], [-l / 2, w / 2, -h / 2], [-l / 2, -w / 2, h / 2], [-l / 2, -w / 2, -h / 2]
    ])

    # 应用旋转
    rot_matrix = np.array([
        [np.cos(rotation), -np.sin(rotation), 0],
        [np.sin(rotation), np.cos(rotation), 0],
        [0, 0, 1]
    ])

    corners_rotated = np.dot(corners, rot_matrix.T)

    # 平移至世界坐标
    corners_rotated[:, 0] += x
    corners_rotated[:, 1] += y
    corners_rotated[:, 2] += z

    return corners_rotated


def calculate_2d_bbox_from_3d(box_3d, calib):
    """从3D边界框计算2D投影边界框"""
    x, y, z = box_3d['location']
    w, l, h = box_3d['dimensions']  # 注意：这里是width, length, height
    rotation = box_3d['rotation']

    # 计算3D框的8个角点
    corners_3d = compute_3d_box_corners(x, y, z, w, l, h, rotation)

    # 投影到图像平面
    corners_2d = calib.project_velo_to_image(corners_3d)

    if corners_2d is not None and len(corners_2d) > 0:
        # 计算2D边界框
        x_min = corners_2d[:, 0].min()
        y_min = corners_2d[:, 1].min()
        x_max = corners_2d[:, 0].max()
        y_max = corners_2d[:, 1].max()

        return [x_min, y_min, x_max, y_max]
    else:
        return [0.0, 0.0, 0.0, 0.0]


def format_kitti_line(box, calib=None):
    """KITTI格式输出"""
    # 计算2D边界框
    bbox_2d = [0.0, 0.0, 0.0, 0.0]
    if calib is not None:
        bbox_2d = calculate_2d_bbox_from_3d(box, calib)

    # 计算alpha角度
    alpha = calculate_alpha(box, calib)

    # 计算截断和遮挡
    truncated = calculate_truncation(bbox_2d, image_size=(640, 480))
    occluded = estimate_occlusion(box)

    # 3D尺寸（注意KITTI格式的顺序：height, width, length）
    height, width, length = box['dimensions'][0], box['dimensions'][1], box['dimensions'][2]

    # 类别映射（将数字类别ID转换为KITTI类别名称）
    class_id = box.get('class', 0)
    if isinstance(class_id, int):
        class_names = ['Car', 'Cyclist', 'Truck']
        class_name = class_names[class_id] if class_id < len(class_names) else 'Car'
    else:
        class_name = class_id

    line = (
        f"{class_name} "  # 1. 类别
        f"{truncated:.2f} "  # 2. 截断程度
        f"{occluded} "  # 3. 遮挡等级
        f"{alpha:.2f} "  # 4. 观测角度
        f"{bbox_2d[0]:.2f} {bbox_2d[1]:.2f} {bbox_2d[2]:.2f} {bbox_2d[3]:.2f} "  # 5-8. 2D边界框
        f"{height:.2f} {width:.2f} {length:.2f} "  # 9-11. 3D尺寸
        f"{box['location'][0]:.2f} {box['location'][1]:.2f} {box['location'][2]:.2f} "  # 12-14. 3D中心坐标
        f"{box['rotation']:.2f} "  # 15. 旋转角
        f"{box['score']:.4f}"  # 16. 检测置信度
    )
    return line


def decode_predictions(predictions, batch_idx, batch_data, data_cfg):
    """解码模型预测为3D边界框"""
    cls_pred = predictions['cls_pred'][batch_idx]  # [C, H, W]
    reg_pred = predictions['reg_pred'][batch_idx]  # [7, H, W]

    # 使用阈值过滤
    cls_probs = torch.sigmoid(cls_pred)
    max_probs, max_indices = torch.max(cls_probs, dim=0)

    pred_boxes = []
    height, width = cls_probs.shape[1:]

    # 获取坐标转换参数
    pc_cfg = data_cfg['pointcloud']
    x_range = pc_cfg['x_range']
    y_range = pc_cfg['y_range']
    grid_h, grid_w = 32, 32
    cell_size_x = (x_range[1] - x_range[0]) / grid_w
    cell_size_y = (y_range[1] - y_range[0]) / grid_h

    # 获取标定数据
    calib = batch_data['calib'][batch_idx] if 'calib' in batch_data else None

    # 遍历BEV网格
    for h in range(height):
        for w in range(width):
            if max_probs[h, w] > 0.3:  # 置信度阈值
                class_id = max_indices[h, w].item()
                reg_data = reg_pred[:, h, w]

                # 将相对坐标转换为绝对坐标
                abs_x = w * cell_size_x + x_range[0] + reg_data[0] * cell_size_x
                abs_y = h * cell_size_y + y_range[0] + reg_data[1] * cell_size_y
                abs_z = reg_data[2]

                # 解码边界框参数
                box = {
                    'class': class_id,
                    'score': max_probs[h, w].item(),
                    'location': [abs_x.item(), abs_y.item(), abs_z.item()],
                    'dimensions': [reg_data[5].item(), reg_data[3].item(), reg_data[4].item()],
                    'rotation': reg_data[6].item()
                }

                # 计算alpha角度
                if calib is not None:
                    box['alpha'] = calculate_alpha(box, calib)
                else:
                    box['alpha'] = 0.0

                pred_boxes.append(box)

    return pred_boxes


def convert_to_kitti_format(predictions, batch_data, data_cfg):
    """将预测结果转换为KITTI格式"""
    results = []
    batch_size = predictions['batch_size']

    for i in range(batch_size):
        sample_id = batch_data['sample_id'][i]
        calib = batch_data['calib'][i] if 'calib' in batch_data else None

        # 解析预测结果
        pred_boxes = decode_predictions(predictions, i, batch_data, data_cfg)

        # 为每个预测框生成KITTI格式行
        for box in pred_boxes:
            kitti_line = format_kitti_line(box, calib)
            results.append({
                'sample_id': sample_id,
                'prediction': kitti_line
            })

    return results


def save_results(results, output_dir):
    """保存结果到文件"""
    # 按样本ID分组
    results_by_id = {}
    for result in results:
        sample_id = result['sample_id']
        if sample_id not in results_by_id:
            results_by_id[sample_id] = []
        results_by_id[sample_id].append(result['prediction'])

    # 保存每个样本的预测结果
    for sample_id, predictions in results_by_id.items():
        output_file = os.path.join(output_dir, f"{sample_id}.txt")

        with open(output_file, 'w') as f:
            for pred_line in predictions:
                f.write(pred_line + '\n')

    print(f"结果已保存到: {output_dir}")


def inference():
    # 加载配置
    with open('../configs/data.yaml', 'r',encoding='utf-8') as f:
        data_cfg = yaml.safe_load(f)

    with open('../configs/model.yaml', 'r',encoding='utf-8') as f:
        model_cfg = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 创建测试数据集
    test_dataset = DRadDataset(data_cfg, split='test')
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_fn
    )

    # 加载模型
    model = RadarCameraFusionModel(model_cfg['model']).to(device)

    checkpoint_path = '../checkpoints/best_model.pth'
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"加载模型检查点: {checkpoint_path}")
        if 'mAP' in checkpoint:
            print(f"训练时最佳 mAP: {checkpoint['mAP']:.4f}")
    else:
        print("警告: 未找到训练好的模型，使用随机初始化权重")

    model.eval()

    # 创建结果目录
    os.makedirs('../results', exist_ok=True)

    # 评估指标
    metrics = DetectionMetrics()

    # 推理和评估
    all_results = []
    all_predictions = []
    all_ground_truths = []

    with torch.no_grad():
        for batch_idx, batch_data in enumerate(test_loader):
            batch_data = move_to_device(batch_data, device)
            predictions = model(batch_data)

            # 转换预测结果为KITTI格式
            kitti_results = convert_to_kitti_format(predictions, batch_data, data_cfg)
            all_results.extend(kitti_results)

            # 收集预测和真值用于评估
            pred_boxes = convert_predictions_to_boxes(predictions, batch_data, data_cfg)
            gt_boxes = convert_labels_to_boxes(batch_data)

            all_predictions.extend(pred_boxes)
            all_ground_truths.extend(gt_boxes)

            # 添加到评估器
            metrics.add_batch(pred_boxes, gt_boxes)

            if (batch_idx + 1) % 100 == 0:
                print(f"已处理 {batch_idx + 1} 个样本")

    # 保存结果
    save_results(all_results, '../results/')
    print(f"推理完成，共处理 {len(all_results)} 个样本")

    # 计算最终评估指标
    print("\n开始计算测试集评估指标...")
    test_results = metrics.evaluate()
    metrics.print_summary()

    # 保存评估结果
    with open('../results/evaluation_results.txt', 'w') as f:
        f.write("3D目标检测测试集评估结果\n")
        f.write("=" * 50 + "\n")
        for cls in metrics.classes:
            result = test_results[cls]
            f.write(f"{cls}: AP={result['ap']:.4f}, Precision={result['precision']:.4f}, "
                    f"Recall={result['recall']:.4f}, F1={result['f1']:.4f}\n")
        f.write(f"mAP: {test_results['mAP']:.4f}\n")

    print(f"评估结果已保存到: ../results/evaluation_results.txt")


if __name__ == '__main__':
    inference()