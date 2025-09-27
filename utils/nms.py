# utils/nms.py
import torch
import numpy as np


def calculate_3d_iou_rotated(box1, boxes2):
    """
    计算3D旋转框的IoU（考虑旋转角度）
    """
    if len(boxes2) == 0:
        return torch.tensor([], device=box1.device)

    # 提取参数
    x1, y1, z1, w1, l1, h1, theta1 = box1
    x2, y2, z2, w2, l2, h2, theta2 = boxes2.unbind(dim=1)

    # 计算轴对齐的3D IoU作为近似
    # 计算每个维度的交集
    inter_x_min = torch.max(x1 - l1 / 2, x2 - l2 / 2)
    inter_x_max = torch.min(x1 + l1 / 2, x2 + l2 / 2)
    inter_y_min = torch.max(y1 - w1 / 2, y2 - w2 / 2)
    inter_y_max = torch.min(y1 + w1 / 2, y2 + w2 / 2)
    inter_z_min = torch.max(z1 - h1 / 2, z2 - h2 / 2)
    inter_z_max = torch.min(z1 + h1 / 2, z2 + h2 / 2)

    # 计算交集体积
    inter_length = torch.clamp(inter_x_max - inter_x_min, min=0)
    inter_width = torch.clamp(inter_y_max - inter_y_min, min=0)
    inter_height = torch.clamp(inter_z_max - inter_z_min, min=0)
    inter_volume = inter_length * inter_width * inter_height

    # 计算并集体积
    volume1 = l1 * w1 * h1
    volume2 = l2 * w2 * h2
    union_volume = volume1 + volume2 - inter_volume

    # 计算角度差异惩罚因子
    angle_diff = torch.abs(theta1 - theta2)
    angle_diff = torch.min(angle_diff, 2 * np.pi - angle_diff)
    angle_similarity = torch.cos(angle_diff)  # 角度相同为1，相差90度为0

    # 结合几何IoU和角度相似性
    iou = (inter_volume / union_volume) * (0.7 + 0.3 * angle_similarity)

    return iou


def rotate_nms_3d(boxes, scores, iou_threshold=0.5):
    """
    修复后的3D旋转框NMS实现
    """
    if len(boxes) == 0:
        return torch.tensor([], dtype=torch.long, device=boxes.device)

    # 按分数降序排序
    sorted_scores, indices = torch.sort(scores, descending=True)
    sorted_boxes = boxes[indices]

    keep = []

    while len(sorted_boxes) > 0:
        # 保留当前最高分框
        current_index = indices[0].item()
        keep.append(current_index)

        if len(sorted_boxes) == 1:
            break

        # 计算当前框与剩余框的IoU
        current_box = sorted_boxes[0]
        other_boxes = sorted_boxes[1:]

        ious = calculate_3d_iou_rotated(current_box, other_boxes)

        # 保留IoU低于阈值的框（需要调整索引）
        keep_mask = ious < iou_threshold

        # 更新剩余框和索引
        sorted_boxes = other_boxes[keep_mask]
        indices = indices[1:][keep_mask]  # 注意：indices[1:]对应other_boxes的索引

    return torch.tensor(keep, dtype=torch.long, device=boxes.device)


class RotatedNMS3D:
    """修复后的3D旋转框NMS类"""

    def __init__(self, iou_threshold=0.5, score_threshold=0.3):
        self.iou_threshold = iou_threshold
        self.score_threshold = score_threshold

    def __call__(self, boxes, scores):
        """
        Args:
            boxes: [N, 7] 张量，每个框为 [x, y, z, w, l, h, theta]
            scores: [N] 张量，每个框的置信度分数
        Returns:
            keep_indices: 保留的框的索引
        """
        if len(boxes) == 0:
            return torch.tensor([], dtype=torch.long, device=boxes.device)

        # 确保输入是张量
        if not isinstance(boxes, torch.Tensor):
            boxes = torch.tensor(boxes, dtype=torch.float32)
        if not isinstance(scores, torch.Tensor):
            scores = torch.tensor(scores, dtype=torch.float32)

        # 过滤低分框
        keep_mask = scores >= self.score_threshold
        filtered_boxes = boxes[keep_mask]
        filtered_scores = scores[keep_mask]

        if len(filtered_boxes) == 0:
            return torch.tensor([], dtype=torch.long, device=boxes.device)

        return rotate_nms_3d(filtered_boxes, filtered_scores, self.iou_threshold)


# 测试函数
def test_nms():
    """测试NMS功能"""
    # 创建测试数据
    boxes = torch.tensor([
        [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0],  # 框1
        [0.1, 0.1, 0.1, 1.0, 1.0, 1.0, 0.0],  # 与框1重叠
        [5.0, 5.0, 5.0, 1.0, 1.0, 1.0, 0.0],  # 不重叠
    ])

    scores = torch.tensor([0.9, 0.8, 0.7])

    nms = RotatedNMS3D(iou_threshold=0.5, score_threshold=0.5)
    keep_indices = nms(boxes, scores)

    print("原始框数量:", len(boxes))
    print("NMS后保留框数量:", len(keep_indices))
    print("保留框索引:", keep_indices)

    # 验证结果
    assert len(keep_indices) == 2  # 应该保留框1和框3
    assert 0 in keep_indices  # 框1应该被保留
    assert 2 in keep_indices  # 框3应该被保留

    print("NMS测试通过!")


if __name__ == "__main__":
    test_nms()