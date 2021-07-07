import os
import torch
import numpy as np
import prettytable as pt
from utils.misc import mkdir_p


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
        for log_name in os.listdir(basic_path + 'checkpoints/periods/'):
            print(log_name)
            if log_name.endswith('result_average'):
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

def takeF1(elem):
    return elem[-5]
        
def summary_stare_fold_result():
    stare_path = 'logs/STARE_fold/'
    logs = []
    f1_sum = 0.
    acc_sum = 0.
    se_sum = 0.
    sp_sum = 0.
    folds = np.array(sorted(list(os.listdir(stare_path))))
    tb = pt.PrettyTable()
    tb.field_names = ['path',
                      'fold',
                      'epochs',
                      'patch_num', 
                      'patch_size',
                      'occlusion',
                      'occ_p',
                      'occ_length',
                      'occ_func',
                      'epoch', 
                      'test_acc', 
                      'specificity', 
                      'sensitivity', 
                      'F1_score']
    for fold in folds:
        max_f1 = 0.
        max_acc = 0.
        max_se = 0.
        max_sp = 0.
        for i in os.listdir(stare_path + fold):
            path = os.path.join(stare_path, fold, i)
            #print(path)
            last_check = torch.load(path + '/checkpoints/last.pt')
            for log in last_check['logs']:
                if log[-2] == last_check['logs'][-1][-1]:
                    if log[6] > max_f1:
                        max_f1 = log[6]
                        max_acc = log[1]
                        max_sp = log[3]
                        max_se = log[4]
                        fold_log = [path, last_check['args']['fold'], last_check['args']['epochs'], last_check['args']['patch_num'], 
                                    last_check['args']['patch_size'], last_check['args']['occlusion'], last_check['args']['occ_p'], 
                                    last_check['args']['occ_length'], last_check['args']['occ_func'], 
                                    log[0],log[1],log[3],log[4],log[6]]
                        
                
        print(fold + ' - f1:' + str(max_f1) + ', acc:' + str(max_acc) + ', sp:' + str(max_sp) + ', se:' + str(max_se))
        f1_sum += max_f1
        acc_sum += max_acc
        sp_sum += max_sp
        se_sum += max_se
        tb.add_row(fold_log)
        
    print('mean f1: ' + str(f1_sum / len(folds)))
    print('mean acc: ' + str(acc_sum / len(folds)))
    print('mean sp: ' + str(sp_sum / len(folds)))
    print('mean se: ' + str(se_sum / len(folds)))
    
    with open('summary/STARE_fold/summary.txt', 'w') as f:
        tb.sortby = 'fold'
        f.write(str(tb) + '\n\n')
        f.write('mean f1: ' + str(f1_sum / len(folds)))
        f.write('\nmean acc: ' + str(acc_sum / len(folds)))
        f.write('\nmean sp: ' + str(sp_sum / len(folds)))
        f.write('\nmean se: ' + str(se_sum / len(folds)))
        
        
def test():
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    stare_path = 'logs/STARE_fold/'
    folds = np.array(sorted(list(os.listdir(stare_path)), reverse=True))
    for fold in folds:
        for i in os.listdir(stare_path + fold):
            last_check = torch.load(os.path.join(stare_path, fold, i) + '/checkpoints/last.pt')
            max_f1 = last_check['logs'][-1][-1]
            path = os.path.join(stare_path, fold, i, 'checkpoints/periods/')
            for check_name in os.listdir(path):
                if check_name.endswith('.pt'):
                    check_path = path + check_name
                    check = torch.load(check_path)
                    print('max_f1: %.4f, cur_f1: %.4f, path: %s' % (max_f1, check['logs'][-1][-2], path))
                    if check['logs'][-1][-2] > max_f1 - 0.015:
                        if os.path.exists(path + check_name[0:-3] + '_result_average/performances.txt') is False:
                            print("================================")
                            print(check_path)
                            print("================================")
                            os.system('/home/chenl/anaconda3/bin/python test_stare.py --check_path ' + check_path + ' --device 3 --batch_size 2')
                        else:
                            performance = open(path + check_name[0:-3] + '_result_average/performances.txt', 'r').read()
                            performance = performance.split('\n')
                            if len(performance) == 0:
                                print("performance ================================")
                                print(check_path)
                                print("================================")
                                os.system('/home/chenl/anaconda3/bin/python test_stare.py --check_path ' + check_path + ' --device 3 --batch_size 2')
                                

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
        for log_name in os.listdir(basic_path + 'checkpoints/periods/'):
            print(log_name)
            if log_name.endswith('result_average'):
                if os.path.exists(basic_path + 'checkpoints/periods/' + log_name + '/performances.txt'):
                    performance = open(basic_path + 'checkpoints/periods/' + log_name + '/performances.txt', 'r').read()
                    print(basic_path + 'checkpoints/periods/' + log_name + '/performances.txt')
                    performance = performance.split('\n')
                    F1_score_avg = round(float(performance[2].split(': ')[1]), 4)
                    Acc_avg = round(float(performance[3].split(': ')[1]), 4)
                    SE_avg = round(float(performance[4].split(': ')[1]), 4)
                    SP_avg = round(float(performance[5].split(': ')[1]), 4)
                    AUC = round(float(performance[0].split(': ')[1]), 4)
                    tb.add_row(param + [
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
    fold_max_logs = []
    mean_F1, mean_acc, mean_SE, mean_SP, mean_AUC = 0., 0., 0., 0., 0.
    for i in os.listdir(path):
        print(i)
        fold_logs = mult_summary_2(path + i + '/')
        logs += fold_logs
        print(i)
        print(fold_logs)
        fold_logs.sort(key=takeF1)
        fold_max_logs.append(fold_logs[-1])
        mean_F1 += fold_logs[-1][-5]
        mean_acc += fold_logs[-1][-4]
        mean_SE += fold_logs[-1][-3]
        mean_SP += fold_logs[-1][-2]
        mean_AUC += fold_logs[-1][-1]
    
    print(fold_max_logs)
    print('mean_F1: %.4f' % (mean_F1 / len(fold_max_logs)))
    print('mean_acc: %.4f' % (mean_acc / len(fold_max_logs)))
    print('mean_SE: %.4f' % (mean_SE / len(fold_max_logs)))
    print('mean_SP: %.4f' % (mean_SP / len(fold_max_logs)))
    print('mean_AUC: %.4f' % (mean_AUC / len(fold_max_logs)))

    with open('summary/' + dataset + '/summary_mean.txt', 'w') as f:
        f.write('mean_F1: %.4f\n' % (mean_F1 / len(fold_max_logs)))
        f.write('mean_acc: %.4f\n' % (mean_acc / len(fold_max_logs)))
        f.write('mean_SE: %.4f\n' % (mean_SE / len(fold_max_logs)))
        f.write('mean_SP: %.4f\n' % (mean_SP / len(fold_max_logs)))
        f.write('mean_AUC: %.4f\n' % (mean_AUC / len(fold_max_logs)))
    
    np.save('summary/' + dataset + '/summary.npy', logs)
    np.save('summary/' + dataset + '/fold_max_logs.npy', fold_max_logs)
    
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
    
    #summary_stare_fold_result()
    # test()
    test_all('logs/STARE/FCN/')
    