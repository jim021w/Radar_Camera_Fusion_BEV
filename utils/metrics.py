# utils/metrics.py
import numpy as np
import torch

class DetectionMetrics:
    """3D目标检测评估指标计算"""
    
    def __init__(self, classes=['Car', 'Cyclist', 'Truck'], iou_threshold=0.5):
        self.classes = classes
        self.iou_threshold = iou_threshold
        self.reset()
    
    def reset(self):
        """重置所有统计量"""
        self.predictions = {cls: [] for cls in self.classes}
        self.ground_truths = {cls: [] for cls in self.classes}
        self.results = {}
    
    def add_batch(self, predictions, ground_truths):
        """
        添加批次预测和真值
        
        Args:
            predictions: 预测列表，每个元素为字典
                [{'class': 'Car', 'bbox_3d': [x,y,z,w,l,h,rot], 'score': 0.9, ...}]
            ground_truths: 真值列表，格式同predictions但不包含score
        """
        for pred in predictions:
            cls = pred['class']
            if cls in self.classes:
                self.predictions[cls].append(pred)
        
        for gt in ground_truths:
            cls = gt['class']
            if cls in self.classes:
                self.ground_truths[cls].append(gt)
    
    def calculate_iou_3d(self, box1, box2):
        """
        计算3D边界框的IoU（使用轴对齐近似）
        
        Args:
            box1: [x, y, z, w, l, h, rot]
            box2: 同上格式
            
        Returns:
            iou: 3D IoU值
        """
        # 提取位置和尺寸（简化：忽略旋转）
        x1, y1, z1 = box1[0], box1[1], box1[2]
        h1, w1, l1 = box1[3], box1[4], box1[5]  # height, width, length

        x2, y2, z2 = box2[0], box2[1], box2[2]
        h2, w2, l2 = box2[3], box2[4], box2[5]  # height, width, length

        # 计算交集区域
        inter_x_min = max(x1 - w1 / 2, x2 - w2 / 2)
        inter_x_max = min(x1 + w1 / 2, x2 + w2 / 2)
        inter_y_min = max(y1 - l1 / 2, y2 - l2 / 2)  # length对应y轴
        inter_y_max = min(y1 + l1 / 2, y2 + l2 / 2)
        inter_z_min = max(z1 - h1 / 2, z2 - h2 / 2)  # height对应z轴
        inter_z_max = min(z1 + h1 / 2, z2 + h2 / 2)
        
        # 检查是否有交集
        if inter_x_min >= inter_x_max or inter_y_min >= inter_y_max or inter_z_min >= inter_z_max:
            return 0.0
        
        # 计算交集和并集体积
        inter_volume = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min) * (inter_z_max - inter_z_min)
        volume1 = w1 * l1 * h1
        volume2 = w2 * l2 * h2
        union_volume = volume1 + volume2 - inter_volume
        
        return inter_volume / union_volume if union_volume > 0 else 0.0
    
    def calculate_ap(self, predictions, ground_truths, iou_threshold=0.5):
        """
        计算单个类别的Average Precision
        
        Args:
            predictions: 该类别的预测列表，按置信度降序排序
            ground_truths: 该类别的真值列表
            iou_threshold: IoU阈值
            
        Returns:
            ap: Average Precision值
        """
        if not predictions:
            return 0.0
        
        # 按置信度排序
        predictions.sort(key=lambda x: x['score'], reverse=True)
        
        # 初始化匹配状态
        gt_matched = [False] * len(ground_truths)
        tp = np.zeros(len(predictions))  # True Positive
        fp = np.zeros(len(predictions))  # False Positive
        
        # 对每个预测进行匹配
        for i, pred in enumerate(predictions):
            best_iou = 0.0
            best_gt_idx = -1
            
            # 寻找最佳匹配的真值框
            for j, gt in enumerate(ground_truths):
                if gt_matched[j]:
                    continue
                
                iou = self.calculate_iou_3d(pred['bbox_3d'], gt['bbox_3d'])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = j
            
            # 判断是否为正样本
            if best_iou >= iou_threshold and best_gt_idx != -1:
                tp[i] = 1
                gt_matched[best_gt_idx] = True
            else:
                fp[i] = 1
        
        # 计算精确率和召回率
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)
        
        recalls = tp_cumsum / max(len(ground_truths), 1)
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-8)
        
        # 平滑精度-召回曲线（Pascal VOC方法）
        for i in range(len(precisions)-2, -1, -1):
            precisions[i] = max(precisions[i], precisions[i+1])
        
        # 计算AP（使用11点插值法）
        ap = 0.0
        for t in np.arange(0, 1.1, 0.1):
            if np.sum(recalls >= t) == 0:
                p = 0
            else:
                p = np.max(precisions[recalls >= t])
            ap += p / 11.0
        
        return ap
    
    def evaluate(self):
        """计算所有类别的评估指标"""
        results = {}
        
        for cls in self.classes:
            predictions = self.predictions[cls]
            ground_truths = self.ground_truths[cls]
            
            if not ground_truths:
                results[cls] = {
                    'ap': 0.0,
                    'precision': 0.0,
                    'recall': 0.0,
                    'f1': 0.0,
                    'num_predictions': len(predictions),
                    'num_ground_truths': 0
                }
                continue
            
            # 计算AP
            ap = self.calculate_ap(predictions, ground_truths, self.iou_threshold)
            
            # 计算精确率、召回率和F1分数
            tp, fp, fn = self.calculate_confusion_matrix(predictions, ground_truths)
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            
            results[cls] = {
                'ap': ap,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'num_predictions': len(predictions),
                'num_ground_truths': len(ground_truths),
                'tp': tp,
                'fp': fp,
                'fn': fn
            }
        
        # 计算mAP
        aps = [results[cls]['ap'] for cls in self.classes]
        mAP = np.mean(aps) if aps else 0.0
        
        results['mAP'] = mAP
        self.results = results
        return results
    
    def calculate_confusion_matrix(self, predictions, ground_truths):
        """计算混淆矩阵"""
        if not predictions:

            return 0, 0, len(ground_truths)


        # 使用贪婪匹配
        gt_matched = [False] * len(ground_truths)
        tp = 0
        
        # 按置信度排序
        predictions.sort(key=lambda x: x['score'], reverse=True)
        
        for pred in predictions:
            best_iou = 0.0
            best_gt_idx = -1
            
            for j, gt in enumerate(ground_truths):
                if gt_matched[j]:
                    continue
                
                iou = self.calculate_iou_3d(pred['bbox_3d'], gt['bbox_3d'])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = j
            
            if best_iou >= self.iou_threshold and best_gt_idx != -1:
                tp += 1
                gt_matched[best_gt_idx] = True
        
        fp = len(predictions) - tp
        fn = len(ground_truths) - tp
        
        return tp, fp, fn
    
    def print_summary(self):
        """打印评估结果摘要"""
        if not self.results:
            self.evaluate()
        
        print("=" * 60)
        print("3D目标检测评估结果")
        print("=" * 60)
        
        for cls in self.classes:
            result = self.results[cls]
            print(f"{cls:>10}: AP={result['ap']:.4f}, "
                  f"Precision={result['precision']:.4f}, "
                  f"Recall={result['recall']:.4f}, "
                  f"F1={result['f1']:.4f}, "
                  f"预测数={result['num_predictions']}, "
                  f"真值数={result['num_ground_truths']}")
        
        print("-" * 60)
        print(f"{'mAP':>10}: {self.results['mAP']:.4f}")
        print("=" * 60)


# 工具函数：数据格式转换
def convert_kitti_to_boxes(kitti_lines):
    """
    将KITTI格式标签转换为边界框字典
    
    Args:
        kitti_lines: KITTI格式的标签行列表
        
    Returns:
        boxes: 边界框字典列表
    """
    boxes = []
    
    for line in kitti_lines:
        if not line.strip():
            continue
            
        parts = line.strip().split()
        if len(parts) < 15:
            continue
            
        box = {
            'class': parts[0],
            'bbox_3d': [
                float(parts[11]),  # x
                float(parts[12]),  # y
                float(parts[13]),  # z
                float(parts[8]),  # height
                float(parts[9]),  # width
                float(parts[10]),  # length
                float(parts[14])   # rotation_y
            ]
        }
        
        # 如果是预测结果，包含置信度
        if len(parts) > 15:
            box['score'] = float(parts[15])
        
        boxes.append(box)
    
    return boxes


# 在训练和推理中使用评估指标
def evaluate_model(model, dataloader, device, metrics):
    """评估模型性能"""
    model.eval()
    metrics.reset()
    
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(dataloader):
            # 模型预测
            predictions = model(batch_data)
            
            # 将预测转换为评估格式
            pred_boxes = convert_predictions_to_boxes(predictions)
            
            # 获取真值
            gt_boxes = convert_labels_to_boxes(batch_data['labels'])
            
            # 添加到评估器
            metrics.add_batch(pred_boxes, gt_boxes)
            
            if (batch_idx + 1) % 10 == 0:
                print(f"已处理 {batch_idx + 1} 个批次")
    
    # 计算最终指标
    results = metrics.evaluate()
    metrics.print_summary()
    
    return results


def convert_predictions_to_boxes(predictions):
    """将模型预测转换为边界框格式"""
    boxes = []
    
    # 这里需要根据你的模型输出格式进行适配
    # 示例实现，需要根据实际模型输出调整
    batch_size = predictions.get('batch_size', 1)
    
    for i in range(batch_size):
        # 解析预测结果
        cls_pred = predictions['cls_pred'][i]  # [C, H, W]
        reg_pred = predictions['reg_pred'][i]  # [7, H, W]
        
        # 使用简单的阈值过滤
        cls_probs = torch.sigmoid(cls_pred)
        max_probs, max_indices = torch.max(cls_probs, dim=0)
        
        height, width = cls_probs.shape[1:]
        
        # 遍历BEV网格生成边界框
        for h in range(height):
            for w in range(width):
                if max_probs[h, w] > 0.3:  # 置信度阈值
                    class_id = max_indices[h, w].item()
                    reg_data = reg_pred[:, h, w]
                    
                    class_names = ['Car', 'Cyclist', 'Truck']
                    if class_id >= len(class_names):
                        continue
                    
                    box = {
                        'class': class_names[class_id],
                        'bbox_3d': [
                            reg_data[0].item(),  # x
                            reg_data[1].item(),  # y
                            reg_data[2].item(),  # z
                            reg_data[5].item(),  # height
                            reg_data[3].item(),  # width
                            reg_data[4].item(),  # length
                            reg_data[6].item()   # rotation
                        ],
                        'score': max_probs[h, w].item()
                    }
                    boxes.append(box)
    
    return boxes


def convert_labels_to_boxes(labels):
    """将标签数据转换为边界框格式"""
    boxes = []
    
    if labels is None:
        return boxes
    
    for label_list in labels:
        if label_list is None:
            continue
            
        for label in label_list:
            box = {
                'class': label['type'],
                'bbox_3d': [
                    label['location'][0],  # x
                    label['location'][1],  # y
                    label['location'][2],  # z
                    label['dimensions'][0],  # height
                    label['dimensions'][1],  # width
                    label['dimensions'][2],  # length
                    label['rotation_y']      # rotation
                ]
            }
            boxes.append(box)
    
    return boxes