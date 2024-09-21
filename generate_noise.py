import numpy as np
from datasets.noise import Simplex_CLASS


# 定义噪声大小
size_w = 608
size_h = 256
noise_size = (256, 608)

# 创建 SimplexNoise 对象
simplexNoise = Simplex_CLASS()

# 生成噪声
simplex_noise = simplexNoise.rand_3d_octaves((3, *noise_size), 6, 0.6)

# 缩放和转置噪声
scaled_noise = 0.2 * simplex_noise.transpose(1, 2, 0)

# 保存噪声到文件
np.save("simplex_noise.npy", scaled_noise)