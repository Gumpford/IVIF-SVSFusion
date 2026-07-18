from __future__ import print_function
import argparse
import os
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
from matplotlib import pyplot as plt
import ssim
from WIT20250526 import Encoder, Decoder
from dataset_1 import fusiondata, MultiEpochsDataLoader
import datetime
def  main():
    parser = argparse.ArgumentParser(description='pix2pix-PyTorch-implementation')
    parser.add_argument('--dataset', type=str, default='aetrain', help='facades')
    parser.add_argument('--batchSize', type=int, default=16, help='training batch size')
    parser.add_argument('--testBatchSize', type=int, default=1, help='testing batch size')
    parser.add_argument('--nEpochs', type=int, default=500, help='number of epochs to train for')
    parser.add_argument('--input_nc', type=int, default=1, help='input image channels')
    parser.add_argument('--output_dim', type=int, default=256, help='input image channels')
    parser.add_argument('--output_nc', type=int, default=1, help='output image channels')
    parser.add_argument('--ngf', type=int, default=64, help='generator filters in first conv layer')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning Rate. Default=0.002')
    # parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
    parser.add_argument('--beta1', type=float, default=0.9, help='beta1 for adam. default=0.5')
    parser.add_argument('--cuda', action='store_true', help='use cuda?')
    parser.add_argument('--threads', type=int, default=20, help='number of threads for data loader to use')
    parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
    parser.add_argument('--lamb', type=int, default=150, help='weight on L1 term in objective')
    parser.add_argument('--A', type=int, default=120, help='weight on L1 term in objective')
    parser.add_argument('--lambda2', type=float, default=2.0, help='Lambda2 value')
    opt = parser.parse_args()

    use_cuda = not opt.cuda and torch.cuda.is_available()
    print(torch.cuda.is_available())

    if opt.cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without --cuda")

    cudnn.benchmark = True

    torch.manual_seed(opt.seed)  #随机种子，参数初始化
    if opt.cuda:
        torch.cuda.manual_seed(opt.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    print('===> Loading datasets')
    root_path = os.path.abspath('./DataSet/TrainData/MSRS/msrs/')

    dataset = fusiondata(root_path)

    training_data_loader = MultiEpochsDataLoader(dataset=dataset, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True, drop_last=True)


    print('===> Building model')
    model1 = Encoder()
    model2 = Decoder()

    model1.train()
    model2.train()

    # 模块初始化
    print('---------- Networks initialized -------------')

    MSE = nn.MSELoss()
    ssim_loss = ssim.ssim
    optimizer1 = optim.AdamW(model1.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999), weight_decay=0.01)  #优化器
    optimizer2 = optim.AdamW(model2.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999), weight_decay=0.01)  # 优化器
    scheduler1 = torch.optim.lr_scheduler.StepLR(optimizer1, step_size=20, gamma=0.5)

    print('-----------------------------------------------')
    if not opt.cuda:
        model1 = model1.to(device)
        model2 = model2.to(device)

    train_loss = []

    def get_current_lr(optimizer):
        return optimizer.param_groups[0]['lr']

    def plot_curve(data):
        fig = plt.figure()
        # x1 = range(0, 10)
        # y1 = loss_list
        plt.plot(range(len(data)), data, color='blue')
        # plt.plot(x1, y1, color='blue')
        plt.legend(['value'], loc='upper right')
        plt.xlabel('number of pictures')
        plt.ylabel('loss curve')
        plt.show()


    def train(epoch):
        total_ssim = 0
        # total_ds_ssim = 0
        global prev_loss
        total_loss = 0
        for iteration, batch in enumerate(training_data_loader, 1):
            imgA_V, imgB_V, imgB_V_E = batch[0], batch[1], batch[2]

            imgA_V = imgA_V.to(device)
            imgB_V = imgB_V.to(device)

            _,_,H,W = imgA_V.shape

            tpA04, tpB04, fA04, fA14, fA24, fB04, fB14, fB24, f_P_ir_16, f_P_ir_32, f_P_ir_64, f_P_vis_16, f_P_vis_32, f_P_vis_64 = model1(imgA_V, imgB_V)
            outputA, outputB, _, _= model2(tpA04, tpB04, f_P_ir_64, f_P_vis_64)

            weighta = imgA_V/((imgA_V+imgB_V)+0.00000000000001)
            weightb = imgB_V/((imgA_V+imgB_V)+0.00000000000001)


            optimizer1.zero_grad()
            optimizer2.zero_grad()

            MSEA = MSE(weighta * outputA, weighta * imgA_V)
            MSEB = MSE(weightb * outputB, weightb * imgB_V)

            ssima = 1 - ssim_loss(outputA, imgA_V)
            ssimb = 1 - ssim_loss(outputB, imgB_V)

            lossMSE = MSEA + MSEB
            ssim = ssima + ssimb
            loss = lossMSE + 5 * ssim

            loss.backward()
            optimizer1.step()
            optimizer2.step()

            train_loss.append(loss.item())
            total_loss += loss.item()
            total_ssim += ssim.item()

        running_loss = total_loss / len(training_data_loader)
        running_ssim = total_ssim / len(training_data_loader)

        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        current_lr = get_current_lr(optimizer1)
        log_info = "===> Epoch[{}]: Loss: {:.4f}, SSIM: {:.4f}, Time: {}, current_lr: {}".format(epoch, running_loss, 1 - running_ssim,
                                                                                 current_time, current_lr)

        print(log_info)

        with open("./parameter/log_{}.txt".format(str(int(opt.lambda2))), "a") as f:
            f.write(log_info + "\n")

        scheduler1.step()
        if optimizer1.param_groups[0]['lr'] <= 1e-6:
            optimizer1.param_groups[0]['lr'] = 1e-6

    def checkpoint(epoch):  #保存参数
        if not os.path.exists("parameter"):
            os.mkdir("parameter")
        if not os.path.exists(os.path.join("parameter", str(int(opt.lambda2)))):
            os.mkdir(os.path.join("parameter", str(int(opt.lambda2))))
        En_model_out_path = "parameter/{}/En_model_epoch_{}.pth".format(str(int(opt.lambda2)), epoch)
        De_model_out_path = "parameter/{}/De_model_epoch_{}.pth".format(str(int(opt.lambda2)), epoch)
        torch.save(model1, En_model_out_path)
        torch.save(model2, De_model_out_path)
        print("Checkpoint saved to {}".format("parameter " + str(int(opt.lambda2))))

    for epoch in range(1, opt.nEpochs + 1):
        train(epoch)
        if epoch % 100 == 0:  #第50轮保存一次参数
            # checkpoint(epoch,List[i])
            checkpoint(epoch)
    plot_curve(train_loss)

    plt.plot(train_loss)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.savefig('./parameter/{}/figure.png'.format(str(int(opt.lambda2))))  # 指定保存路径和文件名


if __name__ == "__main__":
    main()