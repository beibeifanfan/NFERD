import torch.nn as nn
import torch
import torch.nn.functional as F

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class RESAFunction(nn.Module):
    def __init__(self, num_iterations, alpha):
        super(RESAFunction, self).__init__()
        self.num_iterations = num_iterations
        self.alpha = alpha

    def forward(self, feature_mapss):
        """
        Args:
            anomaly_maps (list): A list of anomaly maps from different scales. Each map has shape (batch_size, channels, height, width).

        Returns:
            refined_anomaly_maps (list): A list of refined anomaly maps after RESA processing.
        """
        refined_feature_maps = []
        for feature_maps in feature_mapss:
            refined_map = feature_maps.clone()  # Clone the input anomaly map
            batch_size, channels, height, width = refined_map.size()
            # print(refined_map.size())
            # 通过多次垂直和水平卷积操作，并结合缩放因子，可以逐步细化异常图像，提高其质量。
            for i in range(self.num_iterations):
                # 在每次迭代中，先对异常图像进行垂直卷积操作。
                for direction in ['d', 'u']:
                    conv = nn.Conv2d(channels, channels, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1)).to(device)
                    idx = torch.arange(height)
                    if direction == 'd':
                        idx = (idx + height // 2 ** (self.num_iterations - i)) % height
                    elif direction == 'u':
                        idx = (idx - height // 2 ** (self.num_iterations - i)) % height
                    refined_map.add_(self.alpha * F.relu(conv(refined_map[..., idx, :])))

                # 然后对异常图像进行水平卷积操作。、
                # 对于垂直方向的卷积，根据迭代次数和方向，计算对应的卷积核大小、步幅和填充，然后应用卷积操作并使用ReLU激活函数。
                # 将卷积结果乘以缩放因子alpha并加到原始图像上，以更新异常图像。
                for direction in ['r', 'l']:
                    conv = nn.Conv2d(channels, channels, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0)).to(device)
                    idx = torch.arange(width)
                    if direction == 'r':
                        idx = (idx + width // 2 ** (self.num_iterations - i)) % width
                    elif direction == 'l':
                        idx = (idx - width // 2 ** (self.num_iterations - i)) % width
                    refined_map.add_(self.alpha * F.relu(conv(refined_map[..., idx])))

            refined_feature_maps.append(refined_map)

        return refined_feature_maps
