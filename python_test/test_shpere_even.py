import torch
import matplotlib.pyplot as plt

# 加载 tensor
tensor_model = torch.jit.load("sphere_even_100.pt")  # shape: [100, 3, 1, 1]
tensor = list(tensor_model.parameters())[0]
print(tensor.shape)

# 去掉 shape 中的维度 1

# 拆分为 3 个通道
x, y, z = tensor[:, 0], tensor[:, 1], tensor[:, 2]

# 可视化为 3D 散点图
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x, y, z, s=10, alpha=0.8)

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
plt.title("Tensor[100,3,1,1] Visualization")
plt.savefig("tensor_visualization.png", dpi=300)
print("Visualization complete.")