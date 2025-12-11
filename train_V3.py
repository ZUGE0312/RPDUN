from architecture import *
from utils import *
import torch
import scipy.io as scio
import time
import os
import numpy as np
from torch.autograd import Variable
import datetime
from option_V3 import opt
import torch.nn.functional as F
from collections import OrderedDict
from torch.nn.utils import clip_grad_norm_
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
from torch.optim import Optimizer
import math
from Dataset import Dataset2
import torch.utils.data as data
import torch.multiprocessing as mp

os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
if not torch.cuda.is_available():
    raise Exception('NO GPU!')

# saving path
date_time = str(datetime.datetime.now())
date_time = time2file_name(date_time)
result_path = opt.outf + date_time + f'S{opt.method}/result/'
model_path = opt.outf + date_time + f'S{opt.method}/model/'
if not os.path.exists(result_path):
    os.makedirs(result_path)
if not os.path.exists(model_path):
    os.makedirs(model_path)


model = model_generator(opt.method, opt.pretrained_model_path,opt.iterstage).cuda()
optim_params = model.parameters()
model = torch.nn.DataParallel(model, device_ids=[0,1])
if opt.resuming:
    models1_pretrain = torch.load(os.path.join(opt.resuming_model_path), map_location='cpu')  # .cuda()
    model.load_state_dict(models1_pretrain,strict=True)


# optimizing
optimizer = torch.optim.Adam(optim_params, lr=opt.learning_rate, betas=(0.9, 0.999))
if opt.scheduler=='MultiStepLR':
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=opt.milestones, gamma=opt.gamma)
elif opt.scheduler=='CosineAnnealingLR':
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, opt.max_epoch, eta_min=opt.eta_min)

if opt.loss == 'l1loss':
    criterion = torch.nn.L1Loss().cuda()

elif opt.loss == 'MSEloss':
    criterion = torch.nn.MSELoss().cuda()

elif opt.loss == 'Charbonnierloss':
    criterion = CharbonnierLoss().cuda()

# init mask
mask = sio.loadmat(opt.mask_path)['mask']          # [256, 256]
mask_3d_shift = sio.loadmat(opt.mask3d_path)['mask_3d_shift']      # [256,310,28]

# 数据读取
#train_set = LoadTraining(opt.data_path)
train_dataset = Dataset2(opt, mask, mask_3d_shift, isTrain=True)
train_loader = data.DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers)

#test_set = LoadTest(opt.test_path)
test_dataset = Dataset2(opt, mask, mask_3d_shift)
test_loader = data.DataLoader(test_dataset, batch_size=10, shuffle=False)

def train(epoch, logger):
    epoch_loss = 0
    begin = time.time()
    batch_num = int(np.floor(opt.data_num / opt.batch_size))
    model.train()
    

    if epoch >= opt.resume_epoch:
        for i, (y, HSI, Phi, PhiPhiT) in enumerate(train_loader):
            y, HSI, Phi, PhiPhiT = y.cuda(), HSI.cuda(), Phi.cuda(), PhiPhiT.cuda()

            pred_HSI,llist = model(y, (Phi, PhiPhiT))

            loss = criterion(pred_HSI, HSI)+opt.loss_weight*criterion(torch.mean(HSI,dim=1),llist[-2][:,:,:,:256])
            epoch_loss += loss.data
            optimizer.zero_grad()
            loss.backward()
            if opt.clip_grad:
                clip_grad_norm_(model.parameters(), max_norm=0.2)
            optimizer.step()
        end = time.time()
        lr = optimizer.state_dict()['param_groups'][0]['lr']
        logger.info("===> Epoch {} Complete: Lr: {:.6f} Avg. Loss: {:.6f} time: {:.2f}".
                        format(epoch, lr, epoch_loss / batch_num, (end - begin)))
        scheduler.step()
    else:
        scheduler.step()

    return 0

def test(epoch, logger):
    psnr_list, ssim_list = [], []
    model.eval()
    begin = time.time()
    with torch.no_grad():
        for y, HSI, Phi, PhiPhiT in test_loader:
            y, HSI, Phi, PhiPhiT = y.cuda(), HSI.cuda(), Phi.cuda(), PhiPhiT.cuda()
            model_out,llist = model(y, (Phi, PhiPhiT))

        model_out[model_out<0] = 0
        model_out[model_out>1] = 1

    end = time.time()
    for k in range(model_out.shape[0]):
        psnr_val = torch_psnr(model_out[k, :, :, :], HSI[k, :, :, :])
        ssim_val = torch_ssim(model_out[k, :, :, :], HSI[k, :, :, :])
        psnr_list.append(psnr_val.detach().cpu().numpy())
        ssim_list.append(ssim_val.detach().cpu().numpy())
    pred = np.transpose(model_out.detach().cpu().numpy(), (0, 2, 3, 1)).astype(np.float32)
    truth = np.transpose(HSI.cpu().numpy(), (0, 2, 3, 1)).astype(np.float32)
    psnr_mean = np.mean(np.asarray(psnr_list))
    ssim_mean = np.mean(np.asarray(ssim_list))
    logger.info('===> Epoch {}: testing psnr = {:.2f}, ssim = {:.3f}, time: {:.2f}'
                .format(epoch, psnr_mean, ssim_mean,(end - begin)))

    return pred, truth, psnr_list, ssim_list, psnr_mean, ssim_mean

def args2str(args, indent_l=1):
    """args to string for logger"""
    msg = ''
    for k, v in vars(args).items():
        msg += ' ' * (indent_l * 2) + k + ': ' + str(v) + '\n'
    msg += '\n'
    return msg
def main():
    logger = gen_log(model_path)
    logger.info("Loggers: {}".format(args2str(opt)))
    logger.info("Learning rate:{}, batch_size:{}.\n".format(opt.learning_rate, opt.batch_size))
    logger.info("===> Printing model\n{%s}" % (model))
    params = sum(param.numel() for param in model.parameters())
    logger.info('params: %s' % params)
    psnr_max = 0
    for epoch in range(1, opt.max_epoch + 1):
        #train(epoch, logger)
        (pred, truth, psnr_all, ssim_all, psnr_mean, ssim_mean) = test(epoch, logger)
        if psnr_mean > psnr_max:
            psnr_max = psnr_mean
            if psnr_mean > 28:
                name_img = result_path + '/best.mat'
                #scio.savemat(name_img, {'truth': truth, 'pred': pred, 'psnr_list': psnr_all, 'ssim_list': ssim_all})
                last_best_model_path = checkpoint(model, epoch, model_path, logger)

if __name__ == '__main__':
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    main()


