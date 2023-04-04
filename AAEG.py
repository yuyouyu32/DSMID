import numpy as np
from MyDataLoader import MyDataLoader
from Trainer import Trainer
import os
import json
import torch.utils.data as Data
from utils import Parameters
import click

import warnings
warnings.filterwarnings('ignore')


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
@click.option('--earlystop_patience', type=int,  default=200, help='Early stop mechanism patience epochs')
@click.option('--model_save_path', type=str,  default='./CheckPoints', help='F C D G network storage folder for training')
@click.option('--logs_path', type=str,  default='./Logs', help='Tensorboard log save path')
@click.option('--bucket_result_path', type=str,  default='./bucket_result.json', help='C and D network bucketing results storage path')
def main(batch_size, nz, ngf, ndf, nepochs, lr, beta1, gpu, adv_weight,
         lrd, alpha, earlystop_patience, model_save_path, logs_path, bucket_result_path):
    opt = Parameters(batchSize=batch_size, imageSize=(24, 21), nz=nz, ngf=ngf, ndf=ndf,
                     nepochs=nepochs, lr=lr, beta1=beta1, gpu=gpu, adv_weight=adv_weight,
                     lrd=lrd, alpha=alpha)
    target_names = {
        './Data/HV_O_data.csv': 'Hardness (HV)', './Data/UTS_HT_ALL.csv': 'UTS'}
    img_files = {'./Data/HV_O_data.csv': './Data/Imgs/Schedule_HV.csv',
                 './Data/UTS_HT_ALL.csv': './Data/Imgs/Schedule_HT_ALL.csv'}
    file_s = './Data/HV_O_data.csv'
    model_path = model_save_path
    log_path = logs_path

    if not os.path.isdir(f'{model_path}'):
        os.makedirs(f'{model_path}')
    if not os.path.isdir(f'{log_path}'):
        os.makedirs(f'{log_path}')
    result = {}
    file_t = './Data/UTS_HT_ALL.csv'
    print('-----------------------AAEG Traininig~-----------------------')
    dataloader = MyDataLoader(source_img=img_files[file_s], orginal_data=file_s, targets=target_names[file_s],
                                target_img=img_files[file_t], target_data=file_t, t_target=target_names[file_t])
    source_domain_x, source_domain_y, source_domain_class, target_domain_x, target_domain_y, target_domain_class, original_data_src, n_classes = dataloader.get_dataset(if_norm=True)
    trainer = Trainer(Data.TensorDataset(source_domain_x, source_domain_y, source_domain_class),
                        [target_domain_x, target_domain_y, target_domain_class],
                        opt=opt, nclasses=n_classes, earlystop_patience=earlystop_patience, model_saving_path=model_path,
                        if_init=True)
    best_src_score, best_src_report, best_tgt_score, best_tgt_report = trainer.train(
        batch_size=opt.batchSize, n_epochs=opt.nepochs, Log_path=log_path)
    result = {'source_acc': best_src_score, 'source_report': best_src_report, 'target_acc': best_tgt_score, 'target_report': best_tgt_report}
    json.dump(result, open(bucket_result_path, 'w'), ensure_ascii=False)
    print('-----------------------Train finished!-----------------------')
    print(f'-----------------------Model save in `{model_path}`-----------------------')
    print(f'-----------------------Tensorboard logs save in `{log_path}`-----------------------')
    print(f'-----------------------C and D network bucketing results save in `{bucket_result_path}`-----------------------')

if __name__ == '__main__':
    main()
