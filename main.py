import cv2
import torch
from torchvision import transforms
import os
from mobileposenet import MobilePoseNetV3  

# 初始化模型
model = MobilePoseNetV3(num_keypoints=8)  # 假设每个图像有8个关键点
model.load_state_dict(torch.load('D:/LaWE_weight.pth', map_location=torch.device('cpu')))  # 加载模型权重
model.eval()  # 设置为评估模式

# 图像预处理
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = torch.from_numpy(image).float() / 255.0
    image = image.permute(2, 0, 1)  # 转换为CxHxW
    return image

# 关键点检测
def detect_keypoints(image_path):
    image = preprocess_image(image_path)
    image = transforms.ToTensor()(image).unsqueeze(0)  # 添加批次维度并转换为Tensor
    with torch.no_grad():  # 不计算梯度
        outputs = model(image)  # 模型前向传播
    keypoints = outputs.squeeze(0).numpy()  # 移除批次维度并转换为numpy数组
    return keypoints

# 保存关键点坐标到文件
def save_keypoints_to_file(keypoints, image_file, output_dir):
    keypoints_str = ' '.join([f"{x:.2f},{y:.2f}" for x, y in keypoints[:, :2]]) 
    file_name = os.path.splitext(image_file)[0] + '_keypoints.txt'
    file_path = os.path.join(output_dir, file_name)
    with open(file_path, 'w') as f:
        f.write(keypoints_str)
    print(f'Keypoints saved to {file_path}')

# 定义数据集路径和输出目录
dataset_path = 'D:/weight'
output_dir = os.path.join(dataset_path, 'pyramid_levels')

# 检查输出目录是否存在
if not os.path.exists(output_dir):
    print(f"Error: Directory {output_dir} does not exist.")
else:
    # 获取所有PNG图像文件
    image_files = [f for f in os.listdir(output_dir) if f.lower().endswith('.png')]

    # 处理每个图像文件
    for image_file in image_files:
        image_path = os.path.join(output_dir, image_file)
        keypoints = detect_keypoints(image_path)
        save_keypoints_to_file(keypoints, image_file, output_dir)
