import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import pickle
from datasets.dataset import MVTecDataset_test, MVTecDataset_train, get_data_transforms
from model.resnet import  wide_resnet50_2

class MemoryModule(nn.Module):
    def __init__(self, memory_path='./memory_bank.pkl'):
        super(MemoryModule, self).__init__()
        self.memory_bank = []
        self.memory_path = memory_path
        if os.path.exists(self.memory_path):
            with open(self.memory_path, 'rb') as f:
                self.memory_bank = pickle.load(f)

    def store(self, x):
        # 将输入特征图添加到内存库
        self.memory_bank.append(x.clone().cpu())
        # 将内存库保存到磁盘
        with open(self.memory_path, 'wb') as f:
            pickle.dump(self.memory_bank, f)

    def retrieve_top_k_weighted_sum(self, query_feature, k=20, batch_size=10):
        if not self.memory_bank:
            raise ValueError("Memory bank is empty. Please store some features first.")

        # 将查询特征图展平并移至CPU
        b, c, h, w = query_feature.size()
        query_flat = query_feature.view(b, -1).cpu()  # (b, c * h * w)

        similarities = []
        memory_features = []

        # 从磁盘加载内存库
        with open(self.memory_path, 'rb') as f:
            memory_bank = pickle.load(f)

        # 将内存库中的特征图分批处理
        for i in range(0, len(memory_bank), batch_size):
            batch_memory_features = memory_bank[i:i + batch_size]
            batch_memory_flat = torch.stack([f.view(b, -1) for f in batch_memory_features])

            # 计算余弦相似度
            for j in range(batch_memory_flat.size(0)):
                memory_flat = batch_memory_flat[j]
                sim = F.cosine_similarity(query_flat, memory_flat, dim=1).mean().item()
                similarities.append(sim)
                memory_features.append(memory_bank[i + j])

        # 转换为Tensor并排序
        similarities = torch.tensor(similarities)
        top_k_indices = similarities.topk(k=k).indices  # 获取相似度最高的k个索引

        # 从相似度最高的k个特征中提取对应的特征图
        top_k_memory_features = [memory_features[i] for i in top_k_indices]

        # 将相似度转换为权重
        top_k_similarities = similarities[top_k_indices]
        weights = F.softmax(top_k_similarities, dim=0)  # 将相似度转换为权重，总和为1

        # 计算加权和
        weighted_sum = sum(w * f for w, f in zip(weights, top_k_memory_features))

        return weighted_sum  # 返回加权和特征图

import torch.nn.functional as F

# 示例用法
if __name__ == "__main__":
    classa = 'leather'

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    data_transform, gt_transform = get_data_transforms(256, 256) # (256,608)
    train_path = '/data/xmlg/FJT_RRD/datasets/MVTEC_CLASS_NAMES/'+classa+'/train/'
    train_data = MVTecDataset_train(root=train_path, transform=data_transform)
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=1, shuffle=True)
    encoder, bn = wide_resnet50_2(pretrained=True)
    encoder = encoder.to(device)
    bn = bn.to(device)
    # 初始化三个内存库，每个具有不同的路径
    memory_module1 = MemoryModule(memory_path='./MVTec_memory_bank/memory_banksmall25'+classa+'.pkl').cuda()
    memory_module2 = MemoryModule(memory_path='./MVTec_memory_bank/memory_bankmiddle25'+classa+'.pkl').cuda()
    memory_module3 = MemoryModule(memory_path='./MVTec_memory_bank/memory_bankbig25'+classa+'.pkl').cuda()
    conv11 = nn.Conv2d(1024, 2048, kernel_size=1).to(device)
    conv22 = nn.Conv2d(512, 1024, kernel_size=1).to(device)
    conv33 = nn.Conv2d(256, 512, kernel_size=1).to(device)

    for i, (img, _, _) in enumerate(train_dataloader):
        print(i)
        if i ==25:
            break
        img = img.to(device)
        inputs = encoder(img)
        a = inputs[2]
        a = F.interpolate(a, size=(8,8), mode='bilinear', align_corners=True)
        a = conv11(a)

        b = inputs[1]
        b = F.interpolate(b, size=(16,16), mode='bilinear', align_corners=True)
        b = conv22(b)

        c = inputs[0]
        c = F.interpolate(c, size=(32,32), mode='bilinear', align_corners=True)
        c = conv33(c)

        # 存储特征图到对应的内存库
        memory_module1.store(a) #[2,256,64,152]
        memory_module2.store(b) #[2,512,32,76]
        memory_module3.store(c) #[2,1024,16,38]

    #
    # memory_module = MemoryModule()
    # # 假设我们有一个查询特征图 (1, 256, 256, 256)
    # query_feature = torch.randn(2, 4, 4, 4).cuda()
    # # 假设内存库中有一些特征图
    # memory_feature1 = torch.randn(2, 4, 4, 4).cuda()
    # memory_feature2 = torch.randn(2, 4, 4, 4).cuda()
    # # 将特征图存储到内存库中
    # memory_module.store(memory_feature1)
    # memory_module.store(memory_feature2)
    # # 从内存库中检索最相似的特征图
    # most_similar_feature = memory_module.retrieve_most_similar(query_feature)
    # print("Most similar feature shape:", most_similar_feature.shape)
