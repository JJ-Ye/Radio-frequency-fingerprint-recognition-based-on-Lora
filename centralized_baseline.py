# _*_ coding: utf-8 _*_
# This file is created by C. Zhang for personal use.
# @Time         : 18/08/2022 16:07
# @Author       : tl22089
# @File         : centralized_baseline.py
# @Affiliation  : University of Bristol
import pandas as pd
from tqdm import tqdm
import datetime
import torch
import numpy as np
import random
import os
import h5py
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from utils import get_dataset, AverageMeter,get_Lora_dataset
from options import args_parser
from update import test_inference,test_inference_stft, DatasetSplit,DatasetSplit_stft
from models import MLP, CNN,CNN2D,ResNet18,RFSignalCNN
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.manifold import TSNE
from TripletCNN import CombinedLoss
from torchvision import datasets, transforms

if __name__ == '__main__':
    args = args_parser()
    args.type = 'non-metric'
    seed = 2022
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    device =  'cpu'
    # print(device)
    # define paths
    now = datetime.datetime.now()
    log_name = '../logs/Central-baseline-{:}-{:}-{:}-{:}-{:}-{:}-model-{:}-iid-{:}-bs-{:}/'.format(now.year, now.month,
                                                                                                   now.day,
                                                                                                   now.hour,
                                                                                                   now.minute,
                                                                                                   now.second,
                                                                                                   args.model,
                                                                                                   args.iid,
                                                                                                   args.bs_classes)
    logger = SummaryWriter(log_name)
    # load datasets
    #train_dataset, adapt_dataset, test_dataset, _, _ = get_dataset(args)
    train_dataset, adapt_dataset, test_dataset, train_y,adapt_y, test_y = get_Lora_dataset(args)
    print("***完成数据导入、预处理***")


    # 选择模型
    if args.model == 'cnn':
        # Convolutional neural network
        global_model = CNN(args=args)
    elif args.model == 'cnn2D':
        global_model = CNN2D(args=args)
    elif args.model == 'RFSignalCNN':
        global_model = RFSignalCNN(args=args)
    elif args.model == 'ResNet18':
        global_model = ResNet18(args=args)
    elif args.model == 'mlp':
        # Multi-layer preceptron
        global_model = MLP(args=args)
    else:
        exit('Error: unrecognized model')

    # Set the model to train and send it to device.
    # global_model.load_state_dict(torch.load('../save/centralized.pt'))
    global_model.to(device)
    global_model.train()
    print(global_model)

    # 选择模型优化器
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(global_model.parameters(), lr=args.lr,
                                    momentum=args.momentum)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(global_model.parameters(), lr=args.lr,
                                     weight_decay=1e-4)

    # 定义学习率调度器，监控验证集的损失（test_loss），若损失不再改善则减少学习率
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

    loss_func = torch.nn.CrossEntropyLoss().to(device)   #pytorch的交叉熵函数包括了softmax计算，所有模型没有softmax层
    #loss_func = CombinedLoss(margin=1.0, alpha=0.5)



    # 数据特征和标签分离，返回的内容是一个元组 (image, label)，将图像特征转换为 PyTorch 张量，并将其类型转换为 float，将标签转换为 PyTorch 张量，并将其类型转换为 long
    train_idx = [i for i in range(len(train_dataset))]
    #trainloader = DataLoader(DatasetSplit_lora(train_dataset,train_y, train_idx), batch_size=32, shuffle=True)
    if (args.process == 'RAW_IQ'):
        trainloader = DataLoader(DatasetSplit(train_dataset, train_idx), batch_size=64, shuffle=True)
    if (args.process == 'stft'):
        trainloader = DataLoader(DatasetSplit_stft(train_dataset, train_y, train_idx), batch_size=64, shuffle=True)


    train_loss = AverageMeter()
    train_acc = AverageMeter()
    test_loss = AverageMeter()
    test_acc = AverageMeter()

    history = []
    # 在训练过程中，初始化列表以保存每个epoch的损失和准确率
    train_loss_list = []
    train_accuracy_list = []
    test_loss_list = []
    test_accuracy_list = []

    for epoch in range(args.epochs):
        train_loss_sum = 0
        train_samples = 0

        # 训练阶段
        for batch_idx, (data, labels) in enumerate(trainloader):
            data, labels = data.float().to(device), labels.long().to(device)

            labels = labels.to(device)

            optimizer.zero_grad()
            anchor_features, outputs = global_model(data)

            loss = loss_func(outputs, labels)
            loss.backward()
            optimizer.step()

            # 累计损失
            train_loss_sum += loss.item() * data.size(0)  # 获取当前batch批次的损失值。
            train_samples += data.size(0)

        # 训练阶段
        if (args.process == 'stft'):
            tra_acc, tra_loss, tra_pred, tra_gt = test_inference_stft(args, global_model, train_dataset,train_y)
        if (args.process == 'RAW_IQ'):
            tra_acc, tra_loss, tra_pred, tra_gt = test_inference(args, global_model, train_dataset)
        train_loss_avg = train_loss_sum / train_samples
        train_loss_list.append(train_loss_avg)  # 保存每个epoch的训练损失
        train_accuracy_list.append(tra_acc)  # 保存每个epoch的训练准确率

        # 测试阶段
        if (args.process == 'stft'):
            test_acc, test_loss, _, _ = test_inference_stft(args, global_model, test_dataset,test_y)
        if (args.process == 'RAW_IQ'):
            test_acc, test_loss, _, _ = test_inference(args, global_model, test_dataset)
        test_loss_list.append(test_loss)  # 保存每个epoch的测试损失
        test_accuracy_list.append(test_acc)  # 保存每个epoch的测试准确率

        # 更新学习率调度器
        scheduler.step(test_loss)  # 使用测试集损失来更新学习率

        # 打印日志
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch + 1}/{args.epochs}, Current Learning Rate: {current_lr:.6f}")
        print(f"Train Loss: {tra_loss:.4f}")
        print(f"Train Accuracy: {tra_acc:.2%}")
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test Accuracy: {test_acc:.2%}\n")

        # 记录历史
        history.append((epoch + 1, train_loss_avg, tra_acc, test_loss, test_acc))

    # 保存日志到 DataFrame
    df_log = pd.DataFrame(history, columns=['epoch', 'train_loss_avg', 'tra_acc', 'test_loss', 'test_acc'])
    # 测试最终性能
    if (args.process == 'stft'):
        tra_acc, tra_loss, tra_pred, tra_gt = test_inference_stft(args, global_model, train_dataset, train_y)
    if (args.process == 'RAW_IQ'):
        tra_acc, tra_loss, tra_pred, tra_gt = test_inference(args, global_model, train_dataset)
    print(f"Final Loss on Train: {tra_loss:.4f}")
    print(f"Final Accuracy on Train: {tra_acc:.2%}")
    if (args.process == 'stft'):
        te_acc, te_loss, te_pred, te_gt = test_inference_stft(args, global_model, test_dataset,test_y)
    if (args.process == 'RAW_IQ'):
        te_acc, te_loss, te_pred, te_gt = test_inference(args, global_model, test_dataset)
    print(f"Final Loss on Test: {te_loss:.4f}")
    print(f"Final Accuracy on Test: {te_acc:.2%}")

    # 保存完整模型（结构+权重）
    model_path = '../save/CNNmodel/centralized_model_{:}_{:}_{:}.pth'.format(args.signal, args.model, args.snr_high)
    torch.save(global_model, model_path)

    # 保存权重（仅参数）
    weights_path = '../save/CNNmodel/centralized_weights_{:}_{:}_{:}.pt'.format(args.signal, args.model, args.snr_high)
    torch.save(global_model.state_dict(), weights_path)

    # 保存日志文件（HDF5）
    h5_path = '../save/CNNmodel/centralized_log_{:}_{:}_{:}.h5'.format(args.signal, args.model, args.snr_high)
    with h5py.File(h5_path, 'w') as fo:
        fo.create_dataset('log', data=df_log.values)
        fo.create_dataset('train_pred', data=tra_pred)
        fo.create_dataset('train_label', data=tra_gt)
        fo.create_dataset('test_pred', data=te_pred)
        fo.create_dataset('test_label', data=te_gt)

    # 保存训练和测试损失、准确率到文本文件
    def save_to_txt(file_name, train_loss_list, train_accuracy_list, test_loss_list, test_accuracy_list):
        with open(file_name, 'w') as f:
            f.write('Round Train_Loss Train_Accuracy Test_Loss Test_Accuracy\n')
            for i in range(len(train_loss_list)):
                f.write(
                    f'{i + 1} {train_loss_list[i]} {train_accuracy_list[i]} {test_loss_list[i]} {test_accuracy_list[i]}\n'
                )
    # 定义保存的文件路径
    file_name = '../save/CNNmodel/model[{}]_{}_{}_BT[{}]_pro[{}]_ds[{}]_lr[{}].txt'. \
        format(args.model,args.data_choice, args.epochs, args.Base_or_Tri, args.process,args.datasettype,
                args.lr)
    # 在训练结束后保存数据
    save_to_txt(file_name, train_loss_list, train_accuracy_list, test_loss_list, test_accuracy_list)

    print(f'Loss and accuracy data saved to {file_name}')
    print("模型、权重和日志文件保存完成！")

    #print(f"te_pred shape: {te_pred.shape}")
    #print(f"te_pred: {te_pred}")

    # 生成混淆矩阵
    def plot_confusion_matrix(cm, labels, title='Confusion matrix', cmap=plt.cm.Blues):
        """
        绘制混淆矩阵
        :param cm: 混淆矩阵
        :param labels: 类别标签
        :param title: 图表标题
        :param cmap: 颜色映射
        """
        plt.figure(figsize=(20, 16))
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(labels))
        plt.xticks(tick_marks, labels, rotation=45)
        plt.yticks(tick_marks, labels)

        # 在每个格子里加入数值
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center", color="red")
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.show()

    def calculate_confusion_matrix(predictions, labels, num_classes):
        """
        计算混淆矩阵
        :param predictions: 模型预测标签
        :param labels: 真实标签
        :param num_classes: 类别数
        :return: 归一化混淆矩阵和原始计数矩阵
        """
        # 计算混淆矩阵
        cm = confusion_matrix(labels, predictions, labels=np.arange(num_classes))
        # 归一化混淆矩阵
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        return cm_normalized, cm


    cm = confusion_matrix(te_gt, te_pred)
    # 计算混淆矩阵
    num_classes = len(np.unique(te_gt))  # 假设类别是从0到num_classes-1
    print("num_classes:",num_classes)
    cm_norm, cm_counts = calculate_confusion_matrix(te_pred, te_gt, num_classes)
    # 定义类别标签
    class_labels = [f"Class {i}" for i in range(num_classes)]
    # 绘制归一化混淆矩阵
    plot_confusion_matrix(cm_norm, labels=class_labels, title="Normalized Confusion Matrix")

    # 绘制训练和测试损失曲线
    plt.figure(figsize=(10, 6))
    plt.plot(df_log['epoch'], df_log['train_loss_avg'], label='Train Loss', marker='o')
    plt.plot(df_log['epoch'], df_log['test_loss'], label='Test Loss', marker='s')
    plt.title('Loss Curve', fontsize=16)
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True)
    #plt.savefig('../save/loss_curve_{:}_{:}_{:}.png'.format(args.signal, args.model, args.snr_high))
    plt.show()

    # 绘制测试准确度曲线
    plt.figure(figsize=(10, 6))
    plt.plot(df_log['epoch'], df_log['test_acc'], label='Test Accuracy', color='green', marker='d')
    plt.plot(df_log['epoch'], df_log['tra_acc'], label='Train Accuracy', color='red', marker='o')
    plt.title('Accuracy Curve', fontsize=16)
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Accuracy', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True)
    #plt.savefig('../save/accuracy_curve_{:}_{:}_{:}.png'.format(args.signal, args.model, args.snr_high))
    plt.show()

    # t-SNE可视化样本的模型输出特征分布情况
    def extract_features_and_labels(model, dataloader, device):
        """提取模型的中间层特征向量和对应标签."""
        model.eval()
        features = []
        labels = []
        with torch.no_grad():
            for data, label in tqdm(dataloader, desc="Extracting Features"):
                data = data.float().to(device)
                label = label.to(device)
                feature_vec, _ =  global_model(data)  # 提取特征
                features.append(feature_vec.cpu().numpy())
                labels.append(label.cpu().numpy())
        features = np.vstack(features)
        labels = np.concatenate(labels)
        return features, labels

    #t-SNE可视化输入样本的特征分布情况
    def extract_features_and_labels_2(model, dataloader, device):
        """提取模型的中间层特征向量和对应标签."""
        model.eval()
        features = []
        labels = []
        with torch.no_grad():
            for data, label in tqdm(dataloader, desc="Extracting Features"):
                data = data.float().to(device)
                label = label.to(device)
                feature_vec = data  # 提取特征
                features.append(feature_vec.cpu().numpy())
                labels.append(label.cpu().numpy())
        features = np.vstack(features)
        labels = np.concatenate(labels)
        return features, labels

    def visualize_features_with_tsne(features, labels, class_labels):
        """用 t-SNE 可视化特征向量."""
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        features_2d = tsne.fit_transform(features)

        # 绘图
        plt.figure(figsize=(12, 8))
        scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], c=labels, cmap='tab10', s=15)
        plt.colorbar(scatter, ticks=range(4), label="Class")
        plt.title("t-SNE Visualization of Feature Vectors", fontsize=16)
        plt.xlabel("Dimension 1", fontsize=14)
        plt.ylabel("Dimension 2", fontsize=14)

        # 添加类别标签
        handles, _ = scatter.legend_elements()
        plt.legend(handles, class_labels, title="Classes", fontsize=12)
        plt.show()


    # # 提取特征
    # test_idx = [i for i in range(40000)]
    # testloader = DataLoader(DatasetSplit(test_dataset[:40000], test_idx), batch_size=4, shuffle=False)
    #
    # # 可视化特征向量
    # features, labels = extract_features_and_labels(global_model, testloader, device)
    # print(f"特征向量维度:{features.shape}, 标签维度{labels.shape}")
    # class_labels = [str(i) for i in range(4)]  # 假设类别是 0 到 3
    # visualize_features_with_tsne(features, labels, class_labels)

    # # 可视化原始信号
    # raw_features, raw_labels = extract_features_and_labels_2(global_model, trainloader, device)
    # print(f"原始信号特征向量维度:{raw_features.shape}, 原始标签维度{raw_labels.shape}")
    # raw_class_labels = [str(i) for i in range(4)]  # 假设类别是 0 到 3
    # visualize_features_with_tsne(raw_features, raw_labels, raw_class_labels)

