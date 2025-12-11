import argparse

parser = argparse.ArgumentParser(description="HyperSpectral Image Reconstruction Toolbox")
parser.add_argument('--txt', default='TEST',
                    help='You can set various templates in option.py')

# Hardware specifications
parser.add_argument("--gpu_id", type=str, default='2,3')

# Data specifications
parser.add_argument('--data_root', type=str, default='/home/lkshpc/liumengzu/Dataset/SCI_data', help='dataset directory')

# Saving specifications
parser.add_argument('--outf', type=str, default='./exp/', help='saving_path')

# Model specifications
parser.add_argument('--method', type=str, default='RPDUN', help='method name')
parser.add_argument('--pretrained_model_path', type=str, default=None, help='pretrained model directory')
parser.add_argument('--resuming', type=bool, default=True)
parser.add_argument('--resume_epoch', type=int, default=1)
parser.add_argument('--resuming_model_path', type=str, default='', help='resuming_model_path')

# Training specifications
parser.add_argument('--batch_size', type=int, default=4, help='the number of HSIs per batch')
parser.add_argument("--max_epoch", type=int, default=300, help='total epoch')
parser.add_argument("--scheduler", type=str, default='CosineAnnealingLR', help='MultiStepLR or CosineAnnealingLR')
parser.add_argument("--milestones", type=int, default=[50,100,150,200,250], help='milestones for MultiStepLR')
parser.add_argument("--gamma", type=float, default=0.5, help='learning rate decay for MultiStepLR')
parser.add_argument("--learning_rate", type=float, default=0.0002)
parser.add_argument('--iterstage', type=int, default=5, help='[1,2,3,...,]')
parser.add_argument('--eta_min', type=float, default=1e-6)
parser.add_argument('--loss', type=str, default='Charbonnierloss')
parser.add_argument('--clip_grad', type=bool, default=True)
parser.add_argument('--num_workers', type=int, default=8)
parser.add_argument('--data_num', type=int, default=5000, help='the number of training data')
parser.add_argument('--crop_size', type=int, default=256, help='the size of training data')
parser.add_argument('--noise_level', type=int, default=0, help='the size of training data')
parser.add_argument('--loss_weight', type=float, default=0.01, help='the size of training data')
opt = parser.parse_args()


# dataset
opt.data_path = f"{opt.data_root}/cave_1024_28/"
opt.mask_path = f"{opt.data_root}/mask_simu.mat"
opt.mask3d_path = f"{opt.data_root}/mask_3d_shift.mat"
opt.test_path = f"{opt.data_root}/TSA_simu_data/"

for arg in vars(opt):
    if vars(opt)[arg] == 'True':
        vars(opt)[arg] = True
    elif vars(opt)[arg] == 'False':
        vars(opt)[arg] = False