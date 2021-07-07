import os
import torch
import argparse

parser = argparse.ArgumentParser(description='CNN')
parser.add_argument('--device', type=str, default='0', help='GPU device (default: 0)')
parser.add_argument('--logs_path', type=str, default='logs/', help='load model path')
parser.add_argument('--batch_size', type=int, default=32, help='stride size (default: 1024)')
parser.add_argument('--computer', type=str, default='local', help='load model path')
args = vars(parser.parse_args())
os.environ['CUDA_VISIBLE_DEVICES'] = args['device']
print("os.environ['CUDA_VISIBLE_DEVICES']: ", os.environ['CUDA_VISIBLE_DEVICES'])
#if args['computer'] == 'local':
#    python_path = '/home/izhangh/soft/anaconda3/bin/python '
#else:
#    python_path = '/home/chenl/anaconda3/bin/python '
python_path = '/home/jiangyun/wuchao/MSINet'

def list_all_files(rootdir):
    _files = []
    dirs = os.listdir(rootdir)  # 列出文件夹下所有的目录与文件
    for i in range(0, len(dirs)):
        path = os.path.join(rootdir, dirs[i])
        if os.path.isdir(path):
            _files.extend(list_all_files(path))
        if os.path.isfile(path):
            _files.append(path)
    return _files


for file in list_all_files(args['logs_path']):
    if file.endswith('.pt') and 'periods' in file:
        path_arr = file.split('/')
        basic_path = file[:-len(path_arr[-1])] + path_arr[-1].split('.')[0]
        if os.path.exists(basic_path + '_result_average/performances.txt') is False:
            print("================================")
            print(file)
            print("================================")
            last_check = torch.load(file[:-len(path_arr[-1])-len(path_arr[-2])-1] + '/last.pt')
            max_f1 = last_check['logs'][-1][-1]
            check = torch.load(file)
            print('max_f1: %.4f, cur_f1: %.4f, path: %s' % (max_f1, check['logs'][-1][-2], file))
            if check['logs'][-1][-2] > max_f1 - 0.005:
                os.system(python_path + '3_test.py --check_path "' + file + '" --device ' + args['device'] + ' --batch_size ' + str(args['batch_size']))
