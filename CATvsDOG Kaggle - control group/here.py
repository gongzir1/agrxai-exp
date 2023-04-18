import os
import numpy as np
from PIL import Image

# 定义函数计算余弦相似度
def cosine_similarity(x, y):
    return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))

# 加载数据集
data_folder = './cat_dog'
image_size = 128  # 图像尺寸

# 读取图像文件并将它们转换为向量
images = []
cat_folder = 'cat'
dog_folder = 'dog'

# 读取猫的图像文件
# cat_images = []
cat_folder_path ="./cat-dog/cat"
for file in os.listdir(cat_folder_path):
    filepath = os.path.join(cat_folder_path, file)
    with Image.open(filepath).convert('L') as img:
        # cat_images.append(img)
        img = img.resize((image_size, image_size))
        vector = np.array(img).flatten() / 255
        images.append(vector)


# 读取狗的图像文件
# dog_images = []
dog_folder_path ="./cat-dog/dog"
for file in os.listdir(dog_folder_path):
    filepath = os.path.join(dog_folder_path, file)
    with Image.open(filepath).convert('L') as img:
        # dog_images.append(img)
        img = img.resize((image_size, image_size))
        vector = np.array(img).flatten() / 255
        images.append(vector)
# for subdir, _, files in os.walk(data_folder):
#     for file in files:
#         filepath = os.path.join(subdir, file)
#         with Image.open(filepath).convert('L') as img:
#             img = img.resize((image_size, image_size))
#             vector = np.array(img).flatten() / 255
#             images.append(vector)


# 计算每对图像之间的余弦相似度
similarities = []
for i in range(len(images)):
    for j in range(i+1, len(images)):
        similarity = cosine_similarity(images[i], images[j])
        similarities.append(similarity)

# 计算平均异质性值
mean_heterogeneity = np.mean(1 - np.array(similarities))
print(np.array(similarities))
print('平均异质性值：', mean_heterogeneity)
