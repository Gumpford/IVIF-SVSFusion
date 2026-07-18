from __future__ import print_function
import argparse
import os
from dataset_2 import fusiondata
import torch
from torch.utils.data import DataLoader
from fusenet_0922 import FUSENET  # 64 通道
import torch.backends.cudnn as cudnn
import torch.optim as optim
from matplotlib import pyplot as plt
import datetime
from losses import max_gradint, max_intensity
from Metric_Python.test_function import test_metric

def  main():
    parser = argparse.ArgumentParser(description='pix2pix-PyTorch-implementation')
    parser.add_argument('--dataset', type=str, default='parameter_ceshi', help='facades')
    parser.add_argument('--batchSize', type=int, default=8, help='training batch size')
    parser.add_argument('--nEpochs', type=int, default=500, help='number of epochs to train for')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning Rate. Default=0.002')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
    parser.add_argument('--cuda', action='store_true', help='use cuda?')
    parser.add_argument('--threads', type=int, default=0, help='number of threads for data loader to use')
    parser.add_argument('--seed', type=int, default=12, help='random seed to use. Default=123')
    parser.add_argument('--lamb', type=int, default=150, help='weight on L1 term in objective')
    parser.add_argument('--A', type=int, default=120, help='weight on L1 term in objective')
    parser.add_argument('--lambda2', type=float, default=0.2, help='Lambda2 value')
    device = torch.device("cuda")
    opt = parser.parse_args()
    cudnn.enabled = True
    cudnn.benchmark = True

    torch.manual_seed(opt.seed)
    if opt.cuda:
        torch.cuda.manual_seed(opt.seed)

    print('===> Loading datasets')
    root_path = "D:/DataSet/TrainData/MSRS/msrs/"
    # ae_model_path = 'D:/WVPFusion/parameter/{}/En_model_epoch_500.pth'.format(opt.lambda2)
    ae_model_path = 'D:/WVPFusion/parameter/5/En_model_epoch_500.pth'.format(opt.lambda2)

    NET = torch.load(ae_model_path)
    NET = NET.to(device)
    model2 = FUSENET()
    model2.train()
    model2 = model2.to(device)

    dataset = fusiondata(root_path)
    training_data_loader = DataLoader(dataset=dataset, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True, drop_last=True)
    print('===> Building model')

    # 模块初始化
    print('---------- Networks initialized -------------')
    optimizer1 = optim.AdamW(NET.parameters(), lr=opt.lr/10, betas=(opt.beta1, 0.999), weight_decay=0.01)  #微调
    optimizer2 = optim.AdamW(model2.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999), weight_decay=0.01)

    print('-----------------------------------------------')

    train_loss = []
    metric_qabf = [0]
    def plot_curve(data):
        fig = plt.figure()

        plt.plot(range(len(data)), data, color='blue')
        plt.legend(['value'], loc='upper right')
        plt.xlabel('number of pictures')
        plt.ylabel('loss curve')
    def train(epoch):

        for iteration, batch in enumerate(training_data_loader, 1):
            model2.train()
            NET.train()
            total_batches = 0
            total_loss = 0

            imgA_V, imgB_V, imgB_V_E = batch[0], batch[1], batch[2]
            imgA_V = imgA_V.to(device)
            imgB_V = imgB_V.to(device)
            imgB_V_E = imgB_V_E.to(device)

            _, _, H, W = imgA_V.shape

            weighta = imgA_V/((imgA_V+imgB_V)+0.00000000000001)
            weightb = imgB_V/((imgA_V+imgB_V)+0.00000000000001)

            tpA04, tpB04, fA04, fA14, fA24, fB04, fB14, fB24, f_P_ir_16, f_P_ir_32, f_P_ir_64, f_P_vis_16, f_P_vis_32, f_P_vis_64 = NET(imgA_V, imgB_V)
            F = model2(fA04, fA14, fA24, fB04, fB14, fB24, f_P_ir_16, f_P_ir_32, f_P_ir_64, f_P_vis_16, f_P_vis_32, f_P_vis_64, imgA_V, imgB_V)

            ##############################################################################s算梯度

            max_gradient_detect, out_gradient_detect = max_gradint(imgA_V, imgB_V, F)
            max_image = max_intensity(imgA_V, imgB_V)

            loss_gradient = (out_gradient_detect - max_gradient_detect).norm(1)
            loss_gradient = loss_gradient / (H*W)

            ############################################### 最大亮度损失

            loss_max_intensity = ((F - max_image)).norm(1)
            loss_max_intensity = loss_max_intensity / (H*W)

    #######################################################################################
            loss_norm1 = ((weighta * (F - imgA_V)).norm(1) + (weightb * (F - imgB_V)).norm(1))/(H*W)
            loss = loss_norm1 + 7 * loss_gradient
    #######################################################################################
            #############################################################
            optimizer1.zero_grad()  # 优化器梯度清零
            optimizer2.zero_grad()  # 优化器梯度清零
            loss.backward()
            optimizer1.step()
            optimizer2.step()

            current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # 获取当前时间
            train_loss.append(loss.item())
            total_loss += loss.item()
            total_batches += 1

        avg_loss = total_loss / total_batches
        log_info = "===> Epoch[{}]: Average Loss: {:.4f}, Time: {}".format(epoch, avg_loss, current_time)
        print(log_info)
        if epoch > 400 and epoch % 10 == 0:
            Qabf_mean = test_metric(model2,NET)
            metric_qabf.append(Qabf_mean)
            print(metric_qabf)

            # 保存Qabf_mean到txt文件
            qabf_log_info = "Epoch[{}]: Qabf_mean: {:.4f}".format(epoch, Qabf_mean)
            with open(f"./fuse_parameter/qabf_log_{str(opt.lambda2)}.txt", "a") as f:
                f.write(qabf_log_info + "\n")

            if Qabf_mean >= max(metric_qabf):
                best_model_path = "fuse_parameter/{}/best_model_qabf.pth".format(str(opt.lambda2))
                torch.save(model2, best_model_path)
                print(f"Best Qabf fusion model (epoch {epoch}) saved to {best_model_path}")
                best_encoder_path = "fuse_parameter/{}/best_encoder_qabf.pth".format(str(opt.lambda2))
                torch.save(NET, best_encoder_path)
                print(f"Best Qabf encoder model (epoch {epoch}) saved to {best_encoder_path}")

        with open(r"./fuse_parameter/log_{}.txt".format(str(opt.lambda2)), "a") as f:
            f.write(log_info + "\n")

    def checkpoint(epoch):
        if not os.path.exists("fuse_parameter"):
            os.mkdir("fuse_parameter")
        if not os.path.exists(os.path.join("fuse_parameter", str(opt.lambda2))):
            os.mkdir(os.path.join("fuse_parameter", str(opt.lambda2)))
        net_g_model_out_path = "fuse_parameter/{}/net_model_epoch_{}.pth".format(str(opt.lambda2), epoch)
        torch.save(model2, net_g_model_out_path)
        print("Checkpoint saved to {}".format("fuse_parameter " + str(opt.lambda2)))
        net_g_auto_out_path = "fuse_parameter/{}/net_g_auto_out_path_epoch_{}.pth".format(str(opt.lambda2), epoch)
        torch.save(NET, net_g_auto_out_path)
        print("Checkpoint saved to {}".format("fuse_parameter " + str(opt.lambda2)))

    if __name__ == '__main__':

        for epoch in range(1, opt.nEpochs + 1):
            train(epoch)
            if epoch % 100 == 0:
                checkpoint(epoch)

        plot_curve(train_loss)
        output_dir = './fuse_parameter/{}'.format(str(opt.lambda2))
        save_path = os.path.join(output_dir, 'loss_curve.png')
        plt.savefig(save_path)

    plt.plot(train_loss)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.savefig('./fuse_parameter/{}/figure.png'.format(str(opt.lambda2)))

if __name__ == "__main__":
    main()