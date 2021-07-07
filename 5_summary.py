import os
import torch
import numpy as np
import prettytable as pt
from utils.misc import mkdir_p
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def single_summary(basic_path):
    last_check = torch.load(basic_path + 'checkpoints/last.pt')
    if 'LadderNet' in basic_path:
        from models.LadderNet import LadderNetv6
        net = LadderNetv6(num_classes=2, layers=4, filters=10, inplanes=1)

    net.load_state_dict(last_check['net'], strict=False)
    logs = last_check['logs']
    args = last_check['args']
    summary = open(basic_path + 'summary.txt', 'w')

    for arg in args:
        print(arg + ': ' + str(args[arg]))
        summary.write(arg + ': ' + str(args[arg]) + '\n')

    print('\n\n')
    summary.write('\n\n')

    tb = pt.PrettyTable()
    tb.field_names = [
        'epoch',
        'F1_score',
        'F1_score_avg',
        'Acc_avg',
        'SE_avg',
        'SP_avg',
        'AUC',
        'is_better'
    ]
    not_test = []

    first = []
    second = []
    print(logs[0])
    for log in logs[1:]:
        if os.path.exists(basic_path + 'checkpoints/periods/' + str(log[0]) + '_result_average/performances.txt'):
            performance = open(basic_path + 'checkpoints/periods/' + str(log[0]) + '_result_average/performances.txt', 'r').read()
            performance = performance.split('\n')
            F1_score_avg = round(float(performance[2].split(': ')[1]), 4)
            Acc_avg = round(float(performance[3].split(': ')[1]), 4)
            SE_avg = round(float(performance[4].split(': ')[1]), 4)
            SP_avg = round(float(performance[5].split(': ')[1]), 4)
            AUC = round(float(performance[0].split(': ')[1]), 4)
            is_better = ''
            if F1_score_avg > 0.8202 and SE_avg > 0.7856 and Acc_avg > 0.9561:
                is_better = 'True'
                tb.add_row([
                    int(log[0]),
                    round(float(log[-2]), 4),
                    F1_score_avg,
                    Acc_avg,
                    SE_avg,
                    SP_avg,
                    AUC,
                    is_better
                ])
        else:
            not_test.append(log[0])

    tb.sortby = 'F1_score_avg'
    print(tb)
    print(not_test)
    print(len(not_test))

    summary.write(str(tb) + '\n\n')
    summary.close()

    
def takeF1(elem):
    return elem[-5]
    
    
def mult_summary(sum_path):
    fieldnames = [
        'path',
        'model',
        'batch_size',
        'epochs',
        'lr',
        'threshold',
        'patch_num',
        'patch_size',
        'inside_FOV',
        'occlusion',
        'occ_p',
        'occ_length',
        'occ_func',
        'F1',
        'Acc',
        'SE',
        'SP',
        'AUC',
    ]
    model_name = sum_path.split('/')[-2]
    mkdir_p('summary/'+model_name)
    tb = pt.PrettyTable()
    tb.field_names = fieldnames
    all_logs = []
    for i in os.listdir(sum_path):
        basic_path=sum_path+str(i)+'/'
        print(basic_path)
        last_check = torch.load(basic_path + 'checkpoints/last.pt')
        #if 'LadderNet' in basic_path:
        #    from utils.LadderNetv65 import LadderNetv6
        #    net = LadderNetv6(num_classes=2, layers=4, filters=10, inplanes=1)
            
        #net.load_state_dict(last_check['net'], strict=False)
        logs = last_check['logs']
        args = last_check['args']
        single_logs = []
        param = [
            basic_path,
            args['model'],
            args['batch_size'],
            args['epochs'],
            args['lr'],
            args['threshold_confusion'],
            args['patch_num'],
            args['patch_size'],
            args['inside_FOV'],
            args['occlusion'],
            args['occ_p'],
            int(args['occ_length']),
            args['occ_func'],
        ]
        
        not_test = []
        
        first = []
        second = []
        for log in logs[1:]:
            if os.path.exists(basic_path + 'checkpoints/periods/' + str(log[0]) + '_result_average/performances.txt'):
                performance = open(basic_path + 'checkpoints/periods/' + str(log[0]) + '_result_average/performances.txt', 'r').read()
                print(basic_path + 'checkpoints/periods/' + str(log[0]) + '_result_average/performances.txt')
                performance = performance.split('\n')
                F1_score_avg = round(float(performance[2].split(': ')[1]), 4)
                Acc_avg = round(float(performance[3].split(': ')[1]), 4)
                SE_avg = round(float(performance[4].split(': ')[1]), 4)
                SP_avg = round(float(performance[5].split(': ')[1]), 4)
                AUC = round(float(performance[0].split(': ')[1]), 4)
                single_logs.append(param + [
                    F1_score_avg,
                    Acc_avg,
                    SE_avg,
                    SP_avg,
                    AUC
                ])
                all_logs.append(param + [
                    F1_score_avg,
                    Acc_avg,
                    SE_avg,
                    SP_avg,
                    AUC
                ])
            else:
                not_test.append(log[0])
        
        if len(single_logs) > 1:
            single_logs.sort(key=takeF1)
            tb.add_row(single_logs[-1])
            tb.add_row(single_logs[-2])
            tb.add_row(single_logs[-3])
            tb.add_row(single_logs[-4])
            tb.add_row(single_logs[-5])
    
    with open('summary/'+model_name + '/summary.txt', 'w') as f:
        f.write(str(tb) + '\n\n')
    
    with open('summary/'+model_name+'/summary_f1.txt', 'w') as f:
        tb.sortby = 'F1'
        f.write(str(tb) + '\n\n')
    
    with open('summary/'+model_name+'/summary_occlusion.txt', 'w') as f:
        tb.sortby = 'occlusion'
        tb.reversesort = True
        f.write(str(tb) + '\n\n')
    
    with open('summary/'+model_name+'/summary_occ_length.txt', 'w') as f:
        tb.sortby = 'occ_length'
        tb.reversesort = True
        f.write(str(tb) + '\n\n')
    
    return all_logs

    
def mult_summary_2(sum_path):
    fieldnames = [
        'path',
        'model',
        'batch_size',
        'epochs',
        'lr',
        'threshold',
        'patch_num',
        'patch_size',
        'inside_FOV',
        'data_augmentation',
        'occlusion',
        'occ_p',
        'occ_length',
        'occ_func',
        'epoch',
        'F1',
        'Acc',
        'SE',
        'SP',
        'AUC',
    ]
    data_name = sum_path.split('/')[-3]
    model_name = sum_path.split('/')[-2]
    mkdir_p('summary/'+data_name+'/'+model_name)
    tb = pt.PrettyTable()
    tb.field_names = fieldnames
    all_logs = []
    for i in os.listdir(sum_path):
        basic_path=sum_path+str(i)+'/'
        print(basic_path)
        last_check = torch.load(basic_path + 'checkpoints/last.pt')
        #if 'LadderNet' in basic_path:
        #    from utils.LadderNetv65 import LadderNetv6
        #    net = LadderNetv6(num_classes=2, layers=4, filters=10, inplanes=1)
            
        #net.load_state_dict(last_check['net'], strict=False)
        logs = last_check['logs']
        args = last_check['args']
        if 'data_augmentation' not in args.keys():
            args['data_augmentation'] = False
        single_logs = []
        param = [
            basic_path,
            args['model'],
            args['batch_size'],
            args['epochs'],
            args['lr'],
            args['threshold_confusion'],
            args['patch_num'],
            args['patch_size'],
            args['inside_FOV'],
            args['data_augmentation'],
            args['occlusion'],
            args['occ_p'],
            int(args['occ_length']),
            args['occ_func'],
        ]
        
        not_test = []
        
        first = []
        second = []
        for log_name in os.listdir(basic_path + 'checkpoints/periods/'):
            print(log_name)
            if log_name.endswith('result_average'):
                print(basic_path + 'checkpoints/periods/' + log_name + '/performances.txt')
                if os.path.exists(basic_path + 'checkpoints/periods/' + log_name + '/performances.txt'):
                    performance = open(basic_path + 'checkpoints/periods/' + log_name + '/performances.txt', 'r').read()
                    print(basic_path + 'checkpoints/periods/' + log_name + '/performances.txt')
                    performance = performance.split('\n')
                    F1_score_avg = round(float(performance[2].split(': ')[1]), 4)
                    Acc_avg = round(float(performance[3].split(': ')[1]), 4)
                    SE_avg = round(float(performance[4].split(': ')[1]), 4)
                    SP_avg = round(float(performance[5].split(': ')[1]), 4)
                    AUC = round(float(performance[0].split(': ')[1]), 4)
                    single_logs.append(param + [
                        log_name[0:-15],
                        F1_score_avg,
                        Acc_avg,
                        SE_avg,
                        SP_avg,
                        AUC
                    ])
                    all_logs.append(param + [
                        log_name[0:-15],
                        F1_score_avg,
                        Acc_avg,
                        SE_avg,
                        SP_avg,
                        AUC
                    ])
                else:
                    not_test.append(log[0])
        print(single_logs)
        if len(single_logs) > 1:
            single_logs.sort(key=takeF1)
            tb.add_row(single_logs[-1])
            tb.add_row(single_logs[-2])
            tb.add_row(single_logs[-3])
            tb.add_row(single_logs[-4])
            tb.add_row(single_logs[-5])
    
    with open('summary/'+data_name+'/'+model_name + '/summary.txt', 'w') as f:
        f.write(str(tb) + '\n\n')
    
    with open('summary/'+data_name+'/'+model_name+'/summary_f1.txt', 'w') as f:
        tb.sortby = 'F1'
        f.write(str(tb) + '\n\n')
    
    with open('summary/'+data_name+'/'+model_name+'/summary_occlusion.txt', 'w') as f:
        tb.sortby = 'occlusion'
        tb.reversesort = True
        f.write(str(tb) + '\n\n')
    
    with open('summary/'+data_name+'/'+model_name+'/summary_occ_length.txt', 'w') as f:
        tb.sortby = 'occ_length'
        tb.reversesort = True
        f.write(str(tb) + '\n\n')
    
    return all_logs

    
def test_all(path):
    dataset = path.split('/')[-2]
    fieldnames = [
        'path',
        'model',
        'batch_size',
        'epochs',
        'lr',
        'threshold',
        'patch_num',
        'patch_size',
        'inside_FOV',
        'data_augmentation',
        'occlusion',
        'occ_p',
        'occ_length',
        'occ_func',
        'epoch',
        'F1',
        'Acc',
        'SE',
        'SP',
        'AUC',
    ]
    logs = []
    for i in os.listdir(path):
        print(i)
        logs += mult_summary_2(path + i + '/')
    
    np.save('summary/' + dataset + '/summary.npy', logs)
    
    tb = pt.PrettyTable()
    tb.field_names = fieldnames
    for log in logs:
        tb.add_row(log)    
    
    with open('summary/' + dataset + '/summary.txt', 'w') as f:
        f.write(str(tb) + '\n\n')
    
    with open('summary/' + dataset + '/summary_f1.txt', 'w') as f:
        tb.sortby = 'F1'
        f.write(str(tb) + '\n\n')
    
    with open('summary/' + dataset + '/summary_occlusion.txt', 'w') as f:
        tb.sortby = 'occlusion'
        tb.reversesort = True
        f.write(str(tb) + '\n\n')
    
    with open('summary/' + dataset + '/summary_occ_length.txt', 'w') as f:
        tb.sortby = 'occ_length'
        tb.reversesort = True
        f.write(str(tb) + '\n\n')
    
if __name__ == '__main__':
    #single_summary(basic_path='logs/DRIVE/LadderNet/0/')
    #single_summary(basic_path='logs/DRIVE/LadderNet/1/')
    #test_all('logs/DRIVE/')
    #test_all('logs/CHASE/')
    #test_all('logs/STARE_Old/')
    test_all('/home/sdc_3_7T/jiangyun/lwh/logs/STARE/RENet16/')
    #test_all('logs/DRIVE_M3FCN_op/')
    #test_all('logs/DRIVE_RCS_vs_DA/')
    #test_all('logs/DRIVE_occ_method/')
    #test_all('logs/DRIVE_preprocess/')
