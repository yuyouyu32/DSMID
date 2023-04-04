import torch

import pandas as pd
import numpy as np
import json
import click
from tabulate import tabulate

from utils import Parameters
from models import _netF
from MyDataLoader import MyDataLoader
from edRVFL import EnsembleDeepRVFL

from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_percentage_error

from sklearn.preprocessing import MinMaxScaler, StandardScaler

import warnings
warnings.filterwarnings('ignore')
NORMAL = False
SCALER = MinMaxScaler


def modelHelper(model, x_train, x_test, y_train, y_test, scaler_t):
    model.train(x_train, y_train.reshape(-1), 0)
    y_pred = model.predict(x_test)

    r2 = r2_score(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    if scaler_t:
        y_pred = scaler_t.inverse_transform(y_pred.reshape(-1, 1))
        y_test = scaler_t.inverse_transform(y_test)
    mse = mean_squared_error(y_test, y_pred)
    return r2, mse, mape


def models_run(features_f, y, seeds,  num_nodes, regular_para,
               weight_random_range, bias_random_range, num_layer):
    if NORMAL:
        scaler_f = SCALER()
        features_f = scaler_f.fit_transform(features_f)
        scaler_t = SCALER()
        y = scaler_t.fit_transform(y.reshape(-1, 1))
    else:
        scaler_f, scaler_t = None, None

    kfold = KFold(n_splits=10, shuffle=True,
                  random_state=7).split(features_f, y)
    k_fold_result = {}
    k_fold_result['scores'] = []
    k_fold_result['mse'] = []
    k_fold_result['mape'] = []
    ave_fold = {}
    for index, (train, test) in enumerate(kfold):
        model_edRVFL = EnsembleDeepRVFL(n_nodes=num_nodes, lam=regular_para, w_random_vec_range=weight_random_range,
                                        b_random_vec_range=bias_random_range, activation='relu', n_layer=num_layer, random_seed=seeds[index], same_feature=False, task_type='regression')
        r2, mse, mape = modelHelper(
            model_edRVFL, features_f[train], features_f[test], y[train], y[test], scaler_t)
        k_fold_result['scores'].append(r2)
        k_fold_result['mse'].append(mse)
        k_fold_result['mape'].append(mape)

    ave_fold['ave_score'] = np.mean(k_fold_result['scores'])
    ave_fold['ave_mse'] = np.mean(k_fold_result['mse'])
    ave_fold['ave_mape'] = np.mean(k_fold_result['mape'])
    return ave_fold, k_fold_result


def get_report(opt, num_nodes, regular_para,
               weight_random_range, bias_random_range, num_layer, model_path):
    target_names = {
        './Data/HV_O_data.csv': 'Hardness (HV)', './Data/UTS_HT_ALL.csv': 'UTS', './Data/EL_HT_ALL.csv': 'EL'}
    img_files = {'./Data/HV_O_data.csv': './Data/Imgs/Schedule_HV.csv',
                 './Data/UTS_HT_ALL.csv': './Data/Imgs/Schedule_HT_ALL.csv',
                 './Data/EL_HT_ALL.csv': './Data/Imgs/Schedule_HT_ALL.csv'}

    file_s = './Data/HV_O_data.csv'
    file_t = './Data/UTS_HT_ALL.csv'
    all_result = {}

    # Model

    if torch.cuda.is_available():
        opt.gpu = 1
        model = _netF(opt)
        model.load_state_dict(torch.load(model_path))
    else:
        opt.gpu = -1
        model = _netF(opt)
        model.load_state_dict(torch.load(model_path),
                              map_location=torch.device('cpu'))
    model.double()

    # Data
    dataloader = MyDataLoader(source_img=img_files[file_s], orginal_data=file_s, targets=target_names[file_s],
                              target_img=img_files[file_t], target_data=file_t, t_target=target_names[file_t])
    source_domain_x, source_domain_y, source_domain_class, target_domain_x, target_domain_y, target_domain_class, original_data_src, n_classes = dataloader.get_dataset(
        if_norm=True)

    # Source Domain HV
    print('-----------------HV Training-----------------\n')
    source_domain_y = source_domain_y.numpy()
    out_src = model(source_domain_x)
    feature_src = out_src.cpu().detach().numpy()
    features_o_src = original_data_src.drop(
        columns=[target_names[file_s], 'class']).values
    features_f_src = np.hstack((features_o_src, feature_src))
    ave_result_src, detail_result_src = models_run(features_f_src, source_domain_y, HVSEED,  num_nodes, regular_para,
                                                   weight_random_range, bias_random_range, num_layer)
    all_result['HV'] = {'ave': ave_result_src, 'detail': detail_result_src}
    score_temp = ave_result_src['ave_score']
    print(f'-----------------HV R2 {score_temp}-----------------\n\n')

    # Target Domain UTS and EL
    # UTS
    print('-----------------UTS Training-----------------\n')
    target_domain_y = target_domain_y.numpy()
    out_tgt = model(target_domain_x)
    feature_tgt = out_tgt.cpu().detach().numpy()
    features_o_tgt = pd.read_csv(file_t).drop(
        columns=[target_names[file_t], 'class']).values
    features_f_tgt = np.hstack((features_o_tgt, feature_tgt))
    ave_result_tgt, detail_result_tgt = models_run(features_f_tgt, target_domain_y, UTSSEED,  num_nodes, regular_para,
                                                   weight_random_range, bias_random_range, num_layer)
    all_result['UTS'] = {'ave': ave_result_tgt, 'detail': detail_result_tgt}
    score_temp = ave_result_tgt['ave_score']
    print(f'-----------------HV R2 {score_temp}-----------------\n\n')

    # EL
    print('-----------------EL Training-----------------\n')
    file_EL = './Data/EL_HT_ALL.csv'
    EL_x = pd.read_csv(img_files[file_EL]).astype('double').values
    EL_x = torch.from_numpy(EL_x).reshape(-1, 1, 24, 21)
    EL_data = pd.read_csv(file_EL)
    EL_y = EL_data.loc[:, 'EL'].values
    features_o_EL = EL_data.drop(columns=['EL']).values
    out_EL = model(EL_x)
    feature_EL = out_EL.cpu().detach().numpy()
    features_f_EL = np.hstack((features_o_EL, feature_EL))
    ave_result_EL, detail_result_EL = models_run(features_f_EL, EL_y, ELSEED, num_nodes, regular_para,
                                                 weight_random_range, bias_random_range, num_layer)
    all_result['EL'] = {'ave': ave_result_EL, 'detail': detail_result_EL}
    score_temp = ave_result_EL['ave_score']
    print(f'-----------------HV R2 {score_temp}-----------------\n\n')

    return all_result


HVSEED = [8379, 4384, 4325, 9330, 1658, 2477, 7010, 4081, 2921, 2871]
UTSSEED = [9658, 181, 6133, 5714, 722, 3614, 5065, 6625, 4900, 8080]
ELSEED = [3062, 1903, 3843, 1830, 7819, 5986, 8377, 4332, 189, 9718]


@click.command()
@click.option('--batch_size', type=int, default=512, help='BatchSize of AAEG Training process')
@click.option('--nz', type=int,  default=64, help='Noise size used in G-generated images')
@click.option('--ngf', type=int,  default=64, help='Size of F-network output')
@click.option('--ndf', type=int,  default=64, help='D network extraction feature output size')
@click.option('--nepochs', type=int, default=5000, help='Numbers of training epochs')
@click.option('--lr', type=float,  default=0.00008, help='Learning rate')
@click.option('--beta1', type=float, default=0.8, help='beta1 for adam.')
@click.option('--gpu', type=int, default=1, help='If use GPU for training of testing. 1--used, -1--not used')
@click.option('--adv_weight', type=float, default=1.5, help='weight for adv loss')
@click.option('--lrd', type=float, default=0.0001, help='Learning rate decay value')
@click.option('--alpha', type=float,  default=0.1, help='multiplicative factor for target adv. loss')
@click.option('--num_nodes', type=int, default=128, help='Number of nodes per layer of edRVFL model')
@click.option('--regular_para', type=float, default=1, help='Regularization parameter. of edRVFL model')
@click.option('--weight_random_range', type=int,  default=10, help='Range of random weights')
@click.option('--bias_random_range', type=int, default=10, help='Range of random bias')
@click.option('--num_layer', type=int,  default=32, help='Number of hidden layersNumber of hidden layers')
@click.option('--save_path', type=str,  default=None, help='The predict score result save path')
@click.option('--model_path', type=str,  default='./CheckPoints_Pre/F.pt', help='Pre-Trained F newtwork .pt file path')
def main(batch_size, nz, ngf, ndf, nepochs, lr, beta1, gpu, adv_weight,
         lrd, alpha, num_nodes, regular_para,
         weight_random_range, bias_random_range,
         num_layer, save_path, model_path):
    opt = Parameters(batchSize=batch_size, imageSize=(24, 21), nz=nz, ngf=ngf, ndf=ndf,
                     nepochs=nepochs, lr=lr, beta1=beta1, gpu=gpu, adv_weight=adv_weight,
                     lrd=lrd, alpha=alpha)
    
    weight_random_ranges = [-weight_random_range, weight_random_range]
    bias_random_ranges = [0, bias_random_range]
    result = get_report(opt, num_nodes, regular_para,
                        weight_random_ranges, bias_random_ranges, num_layer, model_path)

    out_result = [ ['HV', result['HV']['ave']['ave_score'], result['HV']['ave']['ave_mape'], result['HV']['ave']['ave_mse']],
    ['UTS',  result['UTS']['ave']['ave_score'], result['UTS']['ave']['ave_mape'], result['UTS']['ave']['ave_mse']],
    ['EL', result['EL']['ave']['ave_score'], result['EL']['ave']['ave_mape'], result['EL']['ave']['ave_mse']]]

    print(tabulate(out_result, headers=['Performance', 'R2', 'MAPE', 'MSE']))
    if save_path:
        with open(save_path, 'w') as f:
            f.write(json.dumps(result))

if __name__ == '__main__':
    main()
