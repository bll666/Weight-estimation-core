import numpy as np
import cv2

# 这里需要根据您具体的应用场景来定义这些点
model_keypoints = np.array([
    [3200.23, 1238.68],  #A
    [1351.11, 1370,94],  # B
    [1527.60, 752.51],  # C
    [1531.78, 2769.24],  # D
    [1801.54, 740.86],  # E
    [1803.19, 1831.87],  # F
])

def ransac_keypoint_alignment(keypoints_list, model_keypoints):
    
    best_inliers = -1
    best_H = None

    for _ in range(100):  # RANSAC迭代次数
        # 随机选择3个点
        rand_idxs = np.random.choice(model_keypoints.shape[0], 4, replace=False)
        points = np.array([keypoints_list[0][rand_idxs[i]] for i in range(4)])
        model_points = model_keypoints[rand_idxs, :]

        # 使用4个点计算单应性矩阵
        H, mask = cv2.findHomography(model_points, points, cv2.RANSAC, 5.0)

        # 计算内点数
        inliers = np.sum(mask)

        if inliers > best_inliers:
            best_inliers = inliers
            best_H = H

    return best_H

def warp_keypoints(keypoints, H):
   
    warped_keypoints = cv2.perspectiveTransform(keypoints[np.newaxis, ...], H)
    return warped_keypoints.squeeze(0)

def bicubic_interpolation(keypoints_list):

    # 计算平均值
    mean_keypoints = np.mean(keypoints_list, axis=0)
    return mean_keypoints

# 关键点列表
detected_keypoints = [...]  

# 使用RANSAC
H = ransac_keypoint_alignment(detected_keypoints, model_keypoints)

# 对每个尺度的关键点进行变换
warped_keypoints = [warp_keypoints(kp, H) for kp in detected_keypoints]

# 双三次插值
final_keypoints = bicubic_interpolation(warped_keypoints)

# 保存最终的关键点坐标
def save_keypoints_to_file(keypoints, output_file):
    with open(output_file, 'w') as f:
        for x, y in keypoints:
            f.write(f"{x:.2f},{y:.2f}\n")

save_keypoints_to_file(final_keypoints, 'final_keypoints.txt')
