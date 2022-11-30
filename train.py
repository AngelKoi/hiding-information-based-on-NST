#coding=gbk
import warnings
warnings.simplefilter("ignore", UserWarning)
import os
import argparse
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.optim import Adam,lr_scheduler
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import torchvision.transforms as transforms
from dataset import PreprocessDataset, denorm
import torch.multiprocessing as mp
from model import Model
import datetime


def weight_init(m):
    if type(m)==nn.Conv2d:
        nn.init.normal(m.weight,mean=0,std=0.5)

def main():
    time_start = datetime.datetime.now()
    parser = argparse.ArgumentParser(description='AdaIN Style Transfer by Pytorch')
    parser.add_argument('--batch_size', '-b', type=int, default=2,
                        help='Number of images in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=100,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--gpu', '-g', type=int, default=0,
                        help='GPU ID(nagative value indicate CPU)')
    parser.add_argument('--learning_rate', '-lr', type=int, default=5e-5,
                        help='learning rate for Adam')
    parser.add_argument('--snapshot_interval', type=int, default=1000,
                        help='Interval of snapshot to generate image')
    parser.add_argument('--train_content_dir', type=str, default='content',
                        help='content images directory for train')
    parser.add_argument('--train_style_dir', type=str, default='style',
                        help='style images directory for train')
    parser.add_argument('--train_secret_dir', type=str, default='secret',
                        help='secret images directory for train')

    parser.add_argument('--test_content_dir', type=str, default='content',
                        help='content images directory for test')
    parser.add_argument('--test_style_dir', type=str, default='style',
                        help='style images directory for test')
    parser.add_argument('--test_secret_dir', type=str, default='secret',
                        help='secret images directory for test')
    parser.add_argument('--save_dir', type=str, default='ckpt2',
                        help='save directory for result and loss')
    parser.add_argument('--reuse', default=None,
                        help='model state path to load for reuse')
    args = parser.parse_args()

    # create directory to save
    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)

    loss_dir = f'{args.save_dir}/loss'
    model_state_dir = f'{args.save_dir}/model_state'
    image_dir = f'{args.save_dir}/image'

    if not os.path.exists(loss_dir):
        os.mkdir(loss_dir)
        os.mkdir(model_state_dir)
        os.mkdir(image_dir)

    # set device on GPU if available, else CPU
    if torch.cuda.is_available() and args.gpu >= 0:
        device = torch.device(f'cuda:{args.gpu}')
        print(f'# CUDA available: {torch.cuda.get_device_name(0)}')
    else:
        device = 'cpu'

    print(f'# Minibatch-size: {args.batch_size}')
    print(f'# epoch: {args.epoch}')
    print('')

    # train_data_transforms=transforms.Compose([
    #     transforms.CenterCrop(256),
    #     transforms.ToTensor(),
    #     transforms.Normalize([0.471, 0.448, 0.408],
    #                         [0.234, 0.239, 0.242])
    # ])
    # prepare dataset and dataLoader
    train_dataset = PreprocessDataset(args.train_content_dir, args.train_style_dir,args.train_secret_dir)
    test_dataset = PreprocessDataset(args.test_content_dir, args.test_style_dir,args.test_secret_dir)
    iters = len(train_dataset)
    print(f'Length of train image pairs: {iters}')

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False,num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,num_workers=2)
    test_iter = iter(test_loader)


    # set model and optimizer2063
    model = Model().to(device)
    #model.apply(weight_init)
    if args.reuse is not None:
        model.load_state_dict(torch.load(args.reuse))
    optimizer_decder = Adam(model.decoder.parameters(), lr=args.learning_rate)
    #lr_sch =lr_scheduler.StepLR(optimizer=optimizer_decder,step_size=100,gamma=0.3)
    optimizer_sec_fea = Adam(model.sec_fea_ext.parameters(), lr=args.learning_rate)
    optimizer_extra = Adam(model.extract.parameters(), lr=1e-3)

    # start training
    loss_list = []
    loss_secret_list = []

    for e in range(1, args.epoch + 1):
        print(f'Start {e} epoch')
        for i, (content, style,secret) in tqdm(enumerate(train_loader, 1)):
            content = content.to(device)
            style = style.to(device)
            secret = secret.to(device)
            loss,loss_secret = model(content, style,secret)
            loss_list.append(loss.item())
            loss_secret_list.append(loss_secret.item())

            if i % 2 == 0:
                optimizer_decder.zero_grad()
                optimizer_sec_fea.zero_grad()
                loss.backward(retain_graph=True)
                optimizer_decder.step()
                optimizer_sec_fea.step()
                #lr_sch.step()
                #print("第%d个epoch的学习率:%f" %(e,optimizer_extra.param_groups[0]['lr']))

            else:
                optimizer_extra.zero_grad()
                loss_secret.backward()
                optimizer_extra.step()
                #optimizer_D.zero_grad()
                # d_loss.backward()
                # optimizer_D.step()

            if args.epoch %10 ==0:
                print(f'[{e}/total {args.epoch} epoch],[{i} /'
                  f'total {round(iters/args.batch_size)} iteration]: {loss.item()}, {loss_secret.item()},') #{d_loss.item()}

            if i % args.snapshot_interval == 0:
                content, style, secret = next(test_iter)
                content = content.to(device)
                style = style.to(device)
                secret = secret.to(device)
                with torch.no_grad():
                    out = model.generate(content, style, secret)
                    out_secret = model.extractor_model(out)


                content = denorm(content, device)
                style = denorm(style, device)
                secret = denorm(secret,device)
                out = denorm(out, device)
                out_secret = denorm(out_secret,device)
                res = torch.cat([content, style, out, secret, out_secret], dim=0)
                res = res.to('cpu')
                save_image(res, f'{image_dir}/{e}_epoch_{i}_iteration.png', nrow=args.batch_size)
        torch.save(model.state_dict(), f'{model_state_dir}/{e}_epoch.pth')
    plt.plot(range(len(loss_list)), loss_list)
    plt.xlabel('iteration')
    plt.ylabel('loss')
    plt.title('train loss')
    plt.savefig(f'{loss_dir}/train_loss.png')

    plt.plot(range(len(loss_secret_list)), loss_secret_list)
    plt.xlabel('iteration')
    plt.ylabel('loss')
    plt.title('secret loss')
    plt.savefig(f'{loss_dir}/serect_loss.png')

    with open(f'{loss_dir}/loss_log.txt', 'w') as f:
        for l in loss_list:
            f.write(f'{l}\n')
    print(f'Loss saved in {loss_dir}')


    with open(f'{loss_dir}/loss_secret_log.txt', 'w') as f:
        for l in loss_secret_list:
            f.write(f'{l}\n')
    print(f'Loss_secret saved in {loss_dir}')

if __name__ == '__main__':
    ngpus_per_node = torch.cuda.device_count()
    multiprocessing_distributed = False
    gpu = 0
    world_size = 1
    if ngpus_per_node > 1:
        multiprocessing_distributed = True

    if multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        world_size = ngpus_per_node * world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main, nprocs=ngpus_per_node, args=(ngpus_per_node, world_size, multiprocessing_distributed))
    else:
        main()
