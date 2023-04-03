import torch

import pandas as pd
import numpy as np
import json

from main import Parameters
from models import _netF
from MyDataLoader import MyDataLoader
from edRVFL import EnsembleDeepRVFL

from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_percentage_error

from sklearn.preprocessing import MinMaxScaler, StandardScaler

import warnings
warnings.filterwarnings('ignore')


num_nodes = 128  # Number of enhancement nodes.
regular_para = 1  # Regularization parameter.
weight_random_range = [-10, 10]  # Range of random weights.
bias_random_range = [0, 10]  # Range of random weights.
num_layer = 32  # Number of hidden layersNumber of hidden layers
HVSEED = [8379, 4384, 4325, 9330, 1658, 2477, 7010, 4081, 2921, 2871]
UTSSEED = [9658, 181, 6133, 5714, 722, 3614, 5065, 6625, 4900, 8080]
ELSEED = [3062, 1903, 3843, 1830, 7819, 5986, 8377, 4332, 189, 9718]

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


def models_run(features_f, y, seeds):
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
    for index, train, test in enumerate(kfold):
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


def main():
    opt = Parameters(batchSize=512, imageSize=(24, 21), nz=32, ngf=64, ndf=64,
                     nepochs=5000, lr=0.00008, beta1=0.8, gpu=-1, adv_weight=1.5,
                     lrd=0.0001, alpha=0.1, outf='results')
    target_names = {
        './Data/HV_O_data.csv': 'Hardness (HV)', './Data/UTS_HT_ALL.csv': 'UTS', './Data/EL_HT_ALL.csv': 'EL'}
    img_files = {'./Data/HV_O_data.csv': './Data/Imgs/Schedule_HV.csv', './Dat/UTS_HT_ALL.csv':
                 '../Data/Imgs/Schedule_HT_ALL.csv',  '../Data/New Data/EL_HT_ALL.csv': '../Data/Imgs/Schedule_HT_ALL.csv'}

    file_s = '../Data/HV_O_data.csv'
    file_t = './Data/UTS_HT_ALL.csv'
    all_result = {}

    # Model
   
    if torch.cuda.is_available():
        opt.gpu = 1
        model = _netF(opt)
        model.load_state_dict(torch.load('./CheckPoints/F.pt'))
    else:
        opt.gpu = -1
        model = _netF(opt)
        model.load_state_dict(torch.load('./CheckPoints/F.pt'), map_location=torch.device('cpu'))
    model.double()

    # Data
    dataloader = MyDataLoader(source_img=img_files[file_s], orginal_data=file_s, targets=target_names[file_s],
                              target_img=img_files[file_t], target_data=file_t, t_target=target_names[file_t])
    source_domain_x, source_domain_y, source_domain_class, target_domain_x, target_domain_y, target_domain_class, original_data_src, n_classes = dataloader.get_dataset(
        if_norm=True)

    # Source Domain HV
    source_domain_y = source_domain_y.numpy()
    out_src = model(source_domain_x)
    feature_src = out_src.cpu().detach().numpy()
    features_o_src = original_data_src.drop(
        columns=[target_names[file_s], 'class']).values
    features_f_src = np.hstack((features_o_src, feature_src))
    ave_result_src, detail_result_src = models_run(features_f=features_f_src, y=source_domain_y)
    all_result['HV'] = {'ave': ave_result_src, 'detail': detail_result_src}

    # Target Domain UTS and EL
    # UTS
    target_domain_y = target_domain_y.numpy()
    out_tgt = model(target_domain_x)
    feature_tgt = out_tgt.cpu().detach().numpy()
    features_o_tgt = pd.read_csv(file_t).drop(
        columns=[target_names[file_t], 'class']).values
    features_f_tgt = np.hstack((features_o_tgt, feature_tgt))
    ave_result_tgt, detail_result_tgt = models_run(
        features_f=features_f_tgt, y=target_domain_y)
    all_result['UTS'] = {'ave': ave_result_tgt, 'detail': detail_result_tgt}

    # EL
    file_EL = '../Data/New Data/EL_HT_ALL.csv'
    EL_x = pd.read_csv(img_files[file_EL]).astype('double').values
    EL_x = torch.from_numpy(EL_x).reshape(-1, 1, 24, 21)
    EL_data = pd.read_csv(file_EL)
    EL_y = EL_data.loc[:, 'EL'].values
    features_o_EL = EL_data.drop(columns=['EL']).values
    out_EL = model(EL_x)
    feature_EL = out_EL.cpu().detach().numpy()
    features_f_EL = np.hstack((features_o_EL, feature_EL))
    ave_result_EL, detail_result_EL = models_run(
        features_f=features_f_EL, y=EL_y)
    all_result['EL'] = {'ave': ave_result_EL, 'detail': detail_result_EL}

    return all_result

if __name__ == '__main__':
    
    print(main())
    # with open(f'./results/T_D_contact_HT_ALL_RVS.json', 'w') as f:
    #     f.write(json.dumps(all_result))
