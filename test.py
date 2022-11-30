import os
import argparse
from PIL import Image
import torch
from torchvision import transforms
from torchvision.utils import save_image
from model import Model
import datetime


normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

normalize_sec = transforms.Normalize(mean=0.5,std=0.5)

trans = transforms.Compose([transforms.ToTensor(),
                            normalize])

trans_sec = transforms.Compose([transforms.CenterCrop(256),
                            #transforms.Grayscale(num_output_channels=1),
                            transforms.ToTensor(),
                            normalize_sec])


def denorm(tensor, device):
    std = torch.Tensor([0.229, 0.224, 0.225]).reshape(-1, 1, 1).to(device)
    mean = torch.Tensor([0.485, 0.456, 0.406]).reshape(-1, 1, 1).to(device)
    res = torch.clamp(tensor * std + mean, 0, 1)
    return res

def denorm_se(tensor, device):
    std = torch.Tensor([0.5]).reshape(-1, 1, 1).to(device)
    mean = torch.Tensor([0.5]).reshape(-1, 1, 1).to(device)
    res = torch.clamp(tensor * std + mean, 0, 1)
    return res


def main():
    time_start = datetime.datetime.now()
    parser = argparse.ArgumentParser(description='AdaIN Style Transfer by Pytorch')
    parser.add_argument('--content', '-c', type=str, default=None,
                        help='Content image path e.g. content.jpg')
    parser.add_argument('--style', '-s', type=str, default=None,
                        help='Style image path e.g. image.jpg')
    parser.add_argument('--secret', '-se', type=str, default=None,
                        help='secret image path e.g. image.jpg')
    parser.add_argument('--output_name', '-o', type=str, default=None,
                        help='Output path for generated image, no need to add ext, e.g. out')
    parser.add_argument('--alpha', '-a', type=float, default=1,
                        help='alpha control the fusion degree in Adain')
    parser.add_argument('--gpu', '-g', type=int, default=0,
                        help='GPU ID(nagative value indicate CPU)')
    parser.add_argument('--model_state_path', type=str, default='ckpt_gray/model_state/100_epoch.pth',
                        help='save directory for result and loss')

    args = parser.parse_args()

    # set device on GPU if available, else CPU
    if torch.cuda.is_available() and args.gpu >= 0:
        device = torch.device(f'cuda:{args.gpu}')
        print(f'# CUDA available: {torch.cuda.get_device_name(0)}')
    else:
        device = 'cpu'

    # set model
    model=Model()
    #model = torch.nn.parallel.DistributedDataParallel(model)
    if args.model_state_path is not None:
        #model.load_state_dict({k.replace('module.', ''): v for k, v in torch.load(args.model_state_path, map_location=lambda storage, loc: storage).items()})
         model.load_state_dict(torch.load(args.model_state_path, map_location=lambda storage, loc: storage))
    model = model.to(device)
    #读模型，并把模型放入设备

    def generation(C_image,S_image,Secret,i):
        c = Image.open(C_image)
        s = Image.open(S_image)
        secret = Image.open(Secret)
        c_tensor = trans(c).unsqueeze(0).to(device)
        s_tensor = trans(s).unsqueeze(0).to(device)
        secret_tensor = trans_sec(secret).unsqueeze(0).to(device)
        with torch.no_grad():
            out_1 = model.generate(c_tensor, s_tensor,secret_tensor,args.alpha)

            out_2 = model.extractor_model(out_1)
    
        out_1 = denorm(out_1, device)
        out_2 = denorm_se(out_2,device)

        if args.output_name is None:
            c_name = os.path.splitext(os.path.basename(C_image))[0]
            s_name = os.path.splitext(os.path.basename(S_image))[0]
            #output_1 = 'result/'+str(number)+f'{c_name}_styl_{s_name}'
            output_1 = 'result/images' + "%d" % i
            output_2 = 'result_secret/images'+ "%d" % i  #str(number)+f'{c_name}_sty_{s_name}'

        save_image(out_1, output_1+'.png', nrow=1)
        save_image(out_2, output_2 + '.png', nrow=1)

        return c,s,secret, out_1,out_2

    content_list = []
    content_list_png = []
    for i in range(100):
        imagename = "images" + "%d" % i + ".png"
        content_list.append(imagename)

    for i in range(100):
        imagename = "images" + "%d" % i + ".png"
        content_list_png.append(imagename)

    for i in range(100):
        #for j in range(100):n
        # print("content/" + content_list[i])
        generation("content_2/" + content_list[i], "style_2/" + content_list[i],"images/"+content_list_png[i], i)

        #     o = Image.open(f'{args.output_name}.jpg')
            #
            # demo = Image.new('RGB', (c.width * 2, c.height))
            # o = o.resize(c.size)
            # s = s.resize((i // 4 for i in c.size))
            #
            # demo.paste(c, (0, 0))
            # demo.paste(o, (c.width, 0))
            # demo.paste(s, (c.width, c.height - s.height))
            # demo.save(f'{args.output_name}_style_transfer_demo.jpg', quality=95)
            #
            # o.paste(s,  (0, o.height - s.height))
            # o.save(f'{args.output_name}_with_style_image.jpg', quality=95)
            #
            # print(f'result saved into files starting with {args.output_name}')
    time_end = datetime.datetime.now()
    print(time_end - time_start)


if __name__ == '__main__':
    main()
