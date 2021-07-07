import argparse
import datetime
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, f1_score
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torchvision.utils import make_grid
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau

import models
from utils.misc import *
from utils.occlusion import occlusion

model_names = sorted(name for name in models.__dict__ if not name.startswith("__") and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='CNN')
parser.add_argument('--device', type=str, default='0', help='GPU device (default: 0)')
parser.add_argument('--dataset', default='DRIVE', choices=['DRIVE', 'CHASE', 'STARE', 'HRF'])
parser.add_argument('--data_path', type=str, default='/home/sdc_3_7T/jiangyun/wuchao/dataset/retinal_vessels/', help='data path')
parser.add_argument('--model', type=str, default='UNet', choices=model_names,
                    help='model architecture: ' + ' | '.join(model_names) + ' (default: FCN)')
parser.add_argument('--batch_size', type=int, default=100, help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=200, help='number of epochs to train (default: 300)')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--lr_decay', action='store_true', default=False, help='learning rate decay')
parser.add_argument('--threshold_confusion', default=0.5, type=float, help='threshold_confusion')
parser.add_argument('--seed', type=int, default=1234, help='random seed (default: 1234)')

parser.add_argument('--patch_num', type=int, default=10000, help='patchs number (default: 800000)')
parser.add_argument('--patch_size', type=int, default=48, help='patch size (default: 48)')
parser.add_argument('--inside_FOV', action='store_true', default=False,
                    help='select the patches only inside the FOV  (default == False)')

parser.add_argument('--data_augmentation', action='store_true', default=False, help='data augmentation')
parser.add_argument('--occlusion', action='store_true', default=False, help='is add occlusion?')
parser.add_argument('--occ_p', default=0.5, type=float, help='occlusion prob')
parser.add_argument('--occ_length', type=int, default=24, help='length of the occlusion')
parser.add_argument('--occ_func', default='fill_next',
                    choices=['fill_0', 'fill_0_tar', 'fill_R', 'fill_R_tar', 'fill_next', 'fill_next_tar'],
                    help='occ_func')

# use last save model
parser.add_argument('--load_last', action='store_true', default=False, help='load last model')
parser.add_argument('--load_path', type=str, default='/home/sdc_3_7T/jiangyun/wuchao/MSINet/logs/time/', help='load model path')
parser.add_argument('--logs_path', type=str, default='/home/sdc_3_7T/jiangyun/wuchao/MSINet/logs/time', help='load model path')

args = vars(parser.parse_args())
os.environ['CUDA_VISIBLE_DEVICES'] = args['device']
cudnn.benchmark = True
torch.cuda.is_available()
torch.manual_seed(args['seed'])
torch.cuda.manual_seed(args['seed'])

threshold_confusion = args['threshold_confusion']

if str(args['logs_path']).endswith('/') is False:
    args['logs_path'] += '/'

if args['load_path'] is not None and str(args['load_path']).endswith('/') is False:
    args['load_path'] += '/'

if args['load_last'] is False:
    mkdir_p(args['logs_path'] + args['dataset'] + '/' + args['model'] + '/')
    index = np.sort(np.array(os.listdir(args['logs_path'] + args['dataset'] + '/' + args['model'] + '/'), dtype=int))
    index = index.max() + 1 if len(index) > 0 else 1
    basic_path = args['logs_path'] + args['dataset'] + '/' + args['model'] + '/' + str(index) + '/'
    mkdir_p(basic_path)
    args['load_path'] = basic_path
    max_acc, max_F1_score, max_sensitivity = 0., 0., 0.
    cur_epoch = 0
    logs = []
    logs.append(
        ['epoch', 'test_acc', 'max_acc', 'specificity', 'sensitivity', 'max_sensitivity', 'F1_score', 'max_F1_score'])
else:
    basic_path = args['load_path']
    assert os.path.exists(basic_path), '目录不存在'
    assert os.path.isfile(basic_path + 'checkpoints/last.pt'), 'Error: no checkpoint file found!'
    checkpoint = torch.load(basic_path + 'checkpoints/last.pt')
    checkpoint['args']['load_last'] = args['load_last']
    checkpoint['args']['load_path'] = args['load_path']
    args = checkpoint['args']
    max_acc = checkpoint['max_acc']
    max_sensitivity = checkpoint['max_sensitivity']
    max_F1_score = checkpoint['max_F1_score']
    cur_epoch = checkpoint['epoch'] + 1
    logs = checkpoint['logs']
    print('保存模型的最后一次训练结果： %s, 当前训练周期: %4d, ' % (str(logs[-1]), cur_epoch))
    assert cur_epoch < args['epochs'], '已经跑完了，cur_epoch: {}，epochs: {}'.format(cur_epoch, args['epochs'])

print('当前日志目录： ' + basic_path)
mkdir_p(basic_path + 'checkpoints/periods/')
mkdir_p(basic_path + 'tensorboard/')
print(args)
with open(basic_path + 'args.txt', 'w+') as f:
    for arg in args:
        f.write(str(arg) + ': ' + str(args[arg]) + '\n')

vis = get_visdom()

if vis is not None:
    import time

    vis.env = args['dataset'] + '_' + args['model'] + '_' + time.strftime('%Y_%m_%d_%H_%M_%S',
                                                                          time.localtime(time.time()))

net = models.__dict__[args['model']]().cuda()
criterion = nn.CrossEntropyLoss().cuda()
optimizer = optim.Adam(net.parameters(), lr=args['lr'])

if args['load_last'] is True and cur_epoch > 0:
    net.load_state_dict(checkpoint['net'], strict=False)
    print('load path: ' + basic_path + 'checkpoints/last.pt')

# 加载数据集
train_orig_imgs, train_orig_gts, train_orig_masks, test_orig_imgs, test_orig_gts, test_orig_masks = get_orig_datasets(
    args)
test_imgs, test_gts, test_imgs_patches, test_masks_patches = get_testing_patchs(
    test_imgs=test_orig_imgs,
    test_gts=test_orig_gts,
    patch_size=args['patch_size'],
)
test_set = TestDataset(test_imgs_patches)
test_loader = DataLoader(test_set, batch_size=args['batch_size'], shuffle=False, num_workers=0)

# Tensorboard
ts_writer = SummaryWriter(log_dir=basic_path + 'tensorboard/', comment=args['model'])
args_str = ''
for arg in args:
    args_str += str(arg) + ': ' + str(args[arg]) + '<br />'
ts_writer.add_text('args', args_str, cur_epoch)

if args['lr_decay'] is True:
    scheduler = ReduceLROnPlateau(optimizer, 'max', factor=0.5, patience=10, verbose=True)

# For visdom
org_images, org_targets, occ_images, occ_targets, vis_mask, vis_outputs = None, None, None, None, None, None


def train():
    global org_images, org_targets, occ_images, occ_targets, vis_mask, vis_outputs, max_acc, max_F1_score, max_sensitivity
    for epoch in range(cur_epoch, args['epochs']):
        if args['lr_decay'] is True:
            scheduler.step(max_F1_score)
        # train network
        train_loss = 0
        train_imgs_patches, train_masks_patches = get_training_patchs(
            train_imgs=train_orig_imgs,
            train_gts=train_orig_gts,
            patch_size=args['patch_size'],
            patch_num=args['patch_num'],
            inside_FOV=args['inside_FOV']  # select the patches only inside the FOV  (default == False)
        )
        train_set = TrainDataset(train_imgs_patches, train_masks_patches, data_augmentation=args['data_augmentation'])
        train_loader = DataLoader(train_set, batch_size=args['batch_size'], shuffle=True, num_workers=0)
        progress_bar = tqdm(train_loader)
        net.train()
        # 训练开始时间
        starttime = datetime.datetime.now()
        for batch_idx, (inputs, targets) in enumerate(progress_bar):
            # print(inputs.shape, targets.shape)
            progress_bar.set_description('Epoch {}-{}'.format(epoch + 1, args['epochs']))

            if vis is not None and batch_idx % 5 == 0:
                org_images = vis.image(
                    make_grid(inputs.data[0:64], nrow=32, normalize=True, scale_each=True, padding=4, pad_value=1),
                    opts=dict(title='Original Images'),
                    win=org_images)
                org_targets = vis.image(
                    make_grid(
                        targets[0:64].type_as(torch.FloatTensor()).view(64, 1, args['patch_size'], args['patch_size']),
                        nrow=32,
                        normalize=True,
                        scale_each=True, padding=4, pad_value=1),
                    opts=dict(title='Original Targets'),
                    win=org_targets)

            if args['occlusion'] is True:
                inputs, targets = occlusion(inputs, targets, args['occ_length'], args['occ_func'], args['occ_p'])
                if vis is not None and batch_idx % 5 == 0:
                    occ_images = vis.image(
                        make_grid(inputs.data[0:64], nrow=32, normalize=True, scale_each=True, padding=4, pad_value=1),
                        opts=dict(title='Occlusion Images'),
                        win=occ_images)
                    occ_targets = vis.image(
                        make_grid(targets[0:64].type_as(torch.FloatTensor()).view(64, 1, args['patch_size'],
                                                                                  args['patch_size']), nrow=32,
                                  padding=4, pad_value=1),
                        opts=dict(title='Occlusion Targets'),
                        win=occ_targets)

            inputs = Variable(inputs.cuda().detach())
            targets = Variable(targets.cuda().detach())

            optimizer.zero_grad()

            output = net(inputs)

            loss = criterion(output, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            progress_bar.set_postfix(loss='%.3f' % (train_loss / (batch_idx + 1)))
            
            # 训练结束时间
        endtime = datetime.datetime.now()
        print('----------time marker---------------')
        print (endtime - starttime)
        print('----------time marker---------------')


        net.eval()
        predictions = []
        with torch.no_grad():
            for batch_idx, test_inputs in enumerate(test_loader):
                test_inputs = test_inputs.detach().cuda()
                test_outputs = net(test_inputs)
                test_outputs = torch.nn.functional.softmax(test_outputs, dim=1)
                test_outputs = test_outputs.permute(0, 2, 3, 1)
                shape = list(test_outputs.shape)
                test_outputs = test_outputs.view(-1, shape[1] * shape[2], 2)
                test_outputs = test_outputs.data.cpu().numpy()
                predictions.append(test_outputs)

        predictions = np.concatenate(predictions, axis=0)
        pred_patches = pred_to_imgs(predictions, args['patch_size'])
        pred_imgs = recompone(pred_patches, test_imgs.shape[2] // args['patch_size'],
                              test_imgs.shape[3] // args['patch_size'])

        if vis is not None:
            vis_mask = vis.image(make_grid(torch.from_numpy(test_gts), nrow=32, normalize=True, scale_each=True),
                                 opts=dict(title='Test Masks'),
                                 win=vis_mask)
            vis_outputs = vis.image(make_grid(torch.from_numpy(pred_imgs), nrow=32, normalize=True, scale_each=True),
                                    opts=dict(title='Test Pred'),
                                    win=vis_outputs)

        y_scores, y_true = pred_only_FOV(pred_imgs, test_gts, test_orig_masks)
        y_pred = np.array([1 if y_scores[i] >= threshold_confusion else 0 for i in range(y_scores.shape[0])])
        confusion = confusion_matrix(y_true, y_pred)
        # print(confusion)
        test_acc = 0
        if float(np.sum(confusion)) != 0:
            test_acc = float(confusion[0, 0] + confusion[1, 1]) / float(np.sum(confusion))
        specificity = 0
        if float(confusion[0, 0] + confusion[0, 1]) != 0:
            specificity = float(confusion[0, 0]) / float(confusion[0, 0] + confusion[0, 1])
        sensitivity = 0
        if float(confusion[1, 1] + confusion[1, 0]) != 0:
            sensitivity = float(confusion[1, 1]) / float(confusion[1, 1] + confusion[1, 0])
        F1_score = f1_score(y_true, y_pred, labels=None, average='binary', sample_weight=None)

        if max_acc < test_acc: max_acc = test_acc
        if max_sensitivity < sensitivity: max_sensitivity = sensitivity
        if max_F1_score < F1_score: max_F1_score = F1_score

        logs.append([epoch, test_acc, max_acc, specificity, sensitivity, max_sensitivity, F1_score, max_F1_score])
        state = {
            'net': net.state_dict(),
            'max_acc': max_acc,
            'max_sensitivity': max_sensitivity,
            'max_F1_score': max_F1_score,
            'epoch': epoch,
            'logs': logs,
            'args': args
        }

        torch.save(state, basic_path + 'checkpoints/periods/{}.pt'.format(epoch))
        torch.save(state, basic_path + 'checkpoints/last.pt')

        ts_writer.add_scalar('train/loss', train_loss / (batch_idx + 1), epoch)
        ts_writer.add_scalar('test/test_acc', test_acc, epoch)
        ts_writer.add_scalar('test/sensitivity', sensitivity, epoch)
        ts_writer.add_scalar('test/F1_score', F1_score, epoch)
        tqdm.write('test acc: {:.4f},     SE: {:.4f},     F1: {:.4f}'.format(test_acc, sensitivity, F1_score))
        tqdm.write(' max acc: {:.4f}, max SE: {:.4f}, max F1: {:.4f}'.format(max_acc, max_sensitivity, max_F1_score))


if __name__ == '__main__':
    train()
    ts_writer.close()
