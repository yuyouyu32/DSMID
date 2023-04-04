from cmath import inf
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as Data
import torch.optim as optim
from sklearn.metrics import accuracy_score, classification_report

import random
import models
import utils
from torch.utils.tensorboard import SummaryWriter


### Set random seed ###
seed = 1104
torch.manual_seed(seed)            # 为CPU设置随机种子
torch.cuda.manual_seed(seed)       # 为当前GPU设置随机种子
torch.cuda.manual_seed_all(seed)   # 为所有GPU设置随机种子
random.seed(seed)
np.random.seed(seed)


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=15, verbose=False, delta=0, path='./CheckPoint', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 15
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, models):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, models)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                self.trace_func(
                    f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, models)
            self.counter = 0

    def save_checkpoint(self, val_loss, models):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(
                f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        for name, model in models.items():
            torch.save(model.state_dict(), f'{self.path}/{name}.pt')
        self.val_loss_min = val_loss
        
    def load_checkpoint(self, models):
        for name, model in models.items():
            model.load_state_dict(torch.load(f'{self.path}/{name}.pt'))


class Trainer:

    def __init__(self, train_data: Data.TensorDataset,
                 #  valid_data: List[torch.Tensor],
                 target_data: List[torch.Tensor],
                 opt, nclasses, earlystop_patience: int, model_saving_path='./CheckPoint/checkpoint.pt',
                 if_init: bool = False) -> None:

        # Defining networks and optimizers
        self.nclasses = nclasses
        self.netG = models._netG(opt, nclasses).double()
        self.netD = models._netD(opt, nclasses).double()
        self.netF = models._netF(opt).double()
        self.netC = models._netC(opt, nclasses).double()
        self.models = {'G': self.netG, 'D': self.netD,
                       'F': self.netF, 'C': self.netC}
        # Weight initialization
        if if_init:
            self.netG.apply(utils.weights_init)
            self.netD.apply(utils.weights_init)
            self.netF.apply(utils.weights_init)
            self.netC.apply(utils.weights_init)

        # Defining loss criterions
        # self.criterion_c = nn.MSELoss()
        self.criterion_c = nn.CrossEntropyLoss()
        self.criterion_s = nn.BCELoss()

        if opt.gpu >= 0:
            self.device = torch.device('cuda:0')
            self.if_cuda = True
            self.netD.cuda()
            self.netG.cuda()
            self.netF.cuda()
            self.netC.cuda()
            self.criterion_c.cuda()
            self.criterion_s.cuda()
            self.criterion_c.cuda()
        else:
            self.device = torch.device('cpu')
            self.if_cuda = False

        # Defining optimizers
        self.optimizerD = optim.Adam(
            self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
        self.optimizerG = optim.Adam(
            self.netG.parameters(), lr=opt.lr * 10, betas=(opt.beta1, 0.999))
        self.optimizerF = optim.Adam(
            self.netF.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
        self.optimizerC = optim.Adam(
            self.netC.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

        # set data
        self.train_data = train_data
        # self.valid_data = [data.to(self.device) for data in valid_data]
        self.target_data = [data.to(self.device) for data in target_data]

        # Other variables
        self.real_label_val = 1
        self.fake_label_val = 0

        self.opt = opt

        # initialize the early_stopping object
        self.early_stopping = EarlyStopping(
            patience=earlystop_patience, verbose=False, path=model_saving_path, trace_func=print)

    def train(self, batch_size: int, n_epochs: int,  Log_path: str):

        best_src_score = -inf
        best_tgt_score = -inf
        best_src_report = None
        best_tgt_report = None
        curr_iter = 0
        writer = SummaryWriter(Log_path, flush_secs=120)

        # Generate train, valid, test data.
        data_iter = Data.DataLoader(self.train_data, batch_size=batch_size,
                                    shuffle=False, sampler=None,
                                    batch_sampler=None, num_workers=0,
                                    collate_fn=None, pin_memory=False,
                                    drop_last=False, timeout=0,
                                    worker_init_fn=None,
                                    multiprocessing_context=None)
        # x_val_pics, y_val, class_val = self.valid_data
        x_target_pics, y_target, class_target = self.target_data

        # Generate real&fake target domain label
        reallabel_t = torch.DoubleTensor(
            len(y_target)).fill_(self.real_label_val)
        fakelabel_t = torch.DoubleTensor(
            len(y_target)).fill_(self.fake_label_val)
        if self.opt.gpu >= 0:
            reallabel_t, fakelabel_t = reallabel_t.cuda(), fakelabel_t.cuda()

        # Training
        for epoch in range(1, n_epochs + 1):
            ###################
            # train the model #
            ###################

            for model in self.models.values():
                model.train()
            tgt_class_losses = []
            tgt_class_scores = []
            c_scores = []
            c_losses = []
            for x_train_pics_batch, y_train_batch, class_train_batch in data_iter:
                # Generate real&fake source domain label
                src_r_f_label_len = len(class_train_batch) if len(
                    class_train_batch) < self.opt.batchSize else self.opt.batchSize
                reallabel_s = torch.DoubleTensor(
                    src_r_f_label_len).fill_(self.real_label_val)
                fakelabel_s = torch.DoubleTensor(
                    src_r_f_label_len).fill_(self.fake_label_val)
                if self.opt.gpu >= 0:
                    reallabel_s, fakelabel_s = reallabel_s.cuda(), fakelabel_s.cuda()

                # Creating one hot vector
                # Source [[1. 0. 0.]
                #         [0. 1. 0.]
                #         [1. 0. 0.]
                #         ...]
                labels_onehot = np.zeros(
                    (src_r_f_label_len, self.nclasses + 1), dtype=np.float64)
                for num in range(src_r_f_label_len):
                    labels_onehot[num, class_train_batch.numpy(
                    ).reshape(-1)[num]] = 1
                src_labels_onehot = torch.from_numpy(labels_onehot)
                # Target [[0. 1. 0.]
                #         [1. 0. 0.]
                #         [0. 1. 0.]
                #         ...]
                labels_onehot = np.zeros(
                    (len(y_target), self.nclasses + 1), dtype=np.float64)
                for num in range(len(y_target)):
                    # labels_onehot[num, self.nclasses] = 1
                    labels_onehot[num, class_target.cpu(
                    ).numpy().reshape(-1)[num]] = 1
                    labels_onehot[num, self.nclasses] = 1
                tgt_labels_onehot = torch.from_numpy(labels_onehot)

                # Set noisy in src input images
                # src_inputs_unnorm = (((x_train_pics_batch*self.std[0]) + self.mean[0]) - 0.5)*2

                # Data to device and Wrapping in variable
                x_train_pics_batch, y_train_batch, class_train_batch = x_train_pics_batch.to(
                    self.device), y_train_batch.to(self.device), class_train_batch.to(self.device)
                # src_inputs_unnorm = src_inputs_unnorm.to(self.device)

                src_labels_onehot = src_labels_onehot.to(self.device)
                tgt_labels_onehot = tgt_labels_onehot.to(self.device)
                ###########################
                # Updates
                ###########################

                # Updating D network
                self.netD.zero_grad()
                # Calculate the result of F using the image of src as input to G
                src_emb = self.netF(x_train_pics_batch)
                src_emb_cat = torch.cat((src_labels_onehot, src_emb), 1)
                src_gen = self.netG(src_emb_cat)
                # Calculate the result of F using the image of tgt as input to G
                tgt_emb = self.netF(x_target_pics)
                tgt_emb_cat = torch.cat((tgt_labels_onehot, tgt_emb), 1)
                tgt_gen = self.netG(tgt_emb_cat)
                # Input the src images with noise into D to get the classification results (task classification and true/false classification), so that D can more easily recognize the real images as real and get the real Label
                src_realoutputD_s, src_realoutputD_c, src_realoutputD_s_f, src_realoutputD_c_f = self.netD(
                    x_train_pics_batch, src_emb)
                errD_src_real_s = self.criterion_s(
                    src_realoutputD_s, reallabel_s)
                errD_src_real_c = self.criterion_c(
                    src_realoutputD_c, class_train_batch)
                errD_src_real_s_f = self.criterion_s(
                    src_realoutputD_s_f, reallabel_s)
                errD_src_real_c_f = self.criterion_c(
                    src_realoutputD_c_f, class_train_batch)
                # Input tgt images with noise into D to get the classification results (task classification and true/false classification), so that D can more easily recognize the real images as real and get the real Label
                tgt_realoutputD_s, tgt_realoutputD_c, tgt_realoutputD_s_f, tgt_realoutputD_c_f = self.netD(
                    x_target_pics, tgt_emb)
                errD_tgt_real_s = self.criterion_s(
                    tgt_realoutputD_s, fakelabel_t)
                errD_tgt_real_c = self.criterion_c(
                    tgt_realoutputD_c, class_target)
                errD_tgt_real_s_f = self.criterion_s(
                    tgt_realoutputD_s_f, fakelabel_t)
                errD_tgt_real_c_f = self.criterion_c(
                    tgt_realoutputD_c_f, class_target)

                # Input the false image generated by G through src's image to D to get the result of classification (true-false classification), so that D can more easily identify the image generated by src as fake
                src_fakeoutputD_s, src_fakeoutputD_c, _, _ = self.netD(
                    src_gen, src_emb)
                errD_src_fake_s = self.criterion_s(
                    src_fakeoutputD_s, fakelabel_s)
                # Input the tgt-generated image into D to get the result of classification (true/false classification), so that D can more easily identify the tgt-generated image as fake
                tgt_fakeoutputD_s, tgt_fakeoutputD_c, _, _ = self.netD(
                    tgt_gen, tgt_emb)
                errD_tgt_fake_s = self.criterion_s(
                    tgt_fakeoutputD_s, fakelabel_t)

                # Loss_D = src image classification task Loss + src_F_emb image classification task + tgt image classification task Loss + tgt_F_emb image classification task + src image true/false classification Loss + src_F_emb image true/false classification Loss + tgt image true/false classification Loss + tgt_F_emb image true/false classification Loss + src generated image true/false classification Loss + tgt generated image true/false classification Loss
                errD = (errD_src_real_c + errD_src_real_c_f) / 2 + (errD_tgt_real_c + errD_tgt_real_c_f) / 2 + (errD_src_real_s +
                                                                                                                errD_src_real_s_f) / 2 + (errD_tgt_real_s + errD_tgt_real_s_f) / 2 + (errD_src_fake_s + errD_tgt_fake_s)
                errD.backward(retain_graph=True)
                self.optimizerD.step()

                # Record D net train update
                writer.add_scalar('D_net/D_src_class_loss',
                                  errD_src_real_c.detach().cpu().item(), curr_iter)
                writer.add_scalar(
                    'D_net/D_src_real_loss',  errD_src_real_s.detach().cpu().item(), curr_iter)
                writer.add_scalar(
                    'D_net/D_src_fake_loss',  errD_src_fake_s.detach().cpu().item(), curr_iter)
                writer.add_scalar(
                    'D_net/D_tgt_fake_loss', errD_tgt_fake_s.detach().cpu().item(), curr_iter)
                writer.add_scalar('D_net/D_tgt_class_loss',
                                  errD_tgt_real_c.detach().cpu().item(), curr_iter)
                writer.add_scalar(
                    'D_net/D_tgt_real_loss',  errD_tgt_real_s.detach().cpu().item(), curr_iter)

                writer.add_scalar('D_net/D_src_class_loss_f',
                                  errD_src_real_c_f.detach().cpu().item(), curr_iter)
                writer.add_scalar(
                    'D_net/D_src_real_loss_f', errD_src_real_s_f.detach().cpu().item(), curr_iter)
                writer.add_scalar(
                    'D_net/D_tgt_class_loss_f', errD_tgt_real_c_f.detach().cpu().item(), curr_iter)
                writer.add_scalar(
                    'D_net/D_tgt_real_loss_f',  errD_tgt_real_s_f.detach().cpu().item(), curr_iter)
                writer.add_scalar('D_net/D_Loss_All',
                                  errD.detach().cpu().item(), curr_iter)

                # Record scores and losses
                _, tgt_real_pred = torch.max(tgt_realoutputD_c_f.data, 1)
                tgt_class_scores.append(accuracy_score(
                    class_target.cpu(), tgt_real_pred.cpu().numpy()))
                tgt_class_losses.append(
                    errD_tgt_real_c_f.detach().cpu().item())
                tgt_report = classification_report(
                    class_target.cpu(), tgt_real_pred.cpu().numpy(), output_dict=True)

                # Updating G network
                self.netG.zero_grad()
                # Input the image generated by src into D
                src_fakeoutputD_s, src_fakeoutputD_c, _, _ = self.netD(
                    src_gen, src_emb)
                # Input the image generated by tgt into D
                tgt_fakeoutputD_s, tgt_fakeoutputD_c, _, _ = self.netD(
                    tgt_gen, tgt_emb)
                # Get the classification output of src fake image, and perform Loss calculation with real Label
                errG_src_c = self.criterion_c(
                    src_fakeoutputD_c, class_train_batch)
                # Get the true and false output of the src fake image, and perform Loss calculation with the true Label
                errG_src_s = self.criterion_s(src_fakeoutputD_s, reallabel_s)
                # Get the classification output of tgt fake images, and perform Loss calculation with real Label
                errG_tgt_c = self.criterion_c(tgt_fakeoutputD_c, class_target)
                # Get the true and false output of the tgt fake image, and perform Loss calculation with the true Label
                errG_tgt_s = self.criterion_s(tgt_fakeoutputD_s, reallabel_t)
                # Loss_G = classification loss of src generated images + discrimination loss of src generated images + classification loss of tgt generated images + discrimination loss of tgy generated images
                errG = errG_src_c + errG_src_s + errG_tgt_c + errG_tgt_s
                errG.backward(retain_graph=True)
                self.optimizerG.step()

                # Record G net train update
                writer.add_scalar('G_net/G_src_gen_class_loss',
                                  errG_src_c.detach().cpu().item(), curr_iter)
                writer.add_scalar('G_net/G_src_fake&r_loss',
                                  errG_src_s.detach().cpu().item(), curr_iter)
                writer.add_scalar('G_net/G_tgt_gen_class_loss',
                                  errG_tgt_c.detach().cpu().item(), curr_iter)
                writer.add_scalar('G_net/G_tgt_fake&r_loss',
                                  errG_tgt_s.detach().cpu().item(), curr_iter)
                writer.add_scalar('G_net/G_Loss_All',
                                  errG.detach().cpu().item(), curr_iter)

                # Updating C network
                self.netC.zero_grad()
                outC = self.netC(src_emb)
                errC = self.criterion_c(outC, class_train_batch)
                _, C_predicted_train = torch.max(outC.data, 1)
                score_train = accuracy_score(
                    class_train_batch.cpu(), C_predicted_train.cpu().numpy())
                # if epoch > 5000: # Try to pre-trained GAN network but not work
                errC.backward(retain_graph=True)
                self.optimizerC.step()
                # Record C net train update
                writer.add_scalar('C_net/accuracy_score',
                                  score_train, curr_iter)
                writer.add_scalar('C_net/Cross_Loss',
                                  errC.detach().cpu().item(), curr_iter)
                # Record scores and losses
                c_scores.append(score_train)
                c_losses.append(errC.detach().cpu().item())
                src_report = classification_report(
                    class_train_batch.cpu(), C_predicted_train.cpu().numpy(), output_dict=True)

                # Updating F network
                # First you need to clear all the gradients left in F, otherwise you will get a very nasty inplace error!!!
                # Note that need to calculate gradients again!!!
                self.netF.zero_grad()
                self.netG.zero_grad()
                self.netC.zero_grad()
                self.netD.zero_grad()
                # Calculate the result of F using the image of src as input to G
                src_emb = self.netF(x_train_pics_batch)
                src_emb_cat = torch.cat((src_labels_onehot, src_emb), 1)
                src_gen = self.netG(src_emb_cat)
                # Calculate the result of F using the image of tgt as input to G
                tgt_emb = self.netF(x_target_pics)
                tgt_emb_cat = torch.cat((tgt_labels_onehot, tgt_emb), 1)
                # Reverse Layer
                alpha = 2. / (1. + np.exp(-10 * float(epoch / n_epochs))) - 1
                tgt_emb_rvs = self.netF(
                    x_target_pics, reverse_grad=True, alpha=alpha)
                tgt_gen = self.netG(tgt_emb_cat)
                # Calculate CORAL Loss(Note that the source and target domains need to have the same amount of data, and random sampling is used here)
                if src_emb.shape[0] < tgt_emb.shape[0]:
                    random_index = torch.LongTensor(random.sample(
                        range(tgt_emb.shape[0]), src_emb.shape[0])).to(self.device)
                    coral_loss = models.CORAL(src_emb, torch.index_select(
                        tgt_emb, 0, random_index), self.device)
                else:
                    random_index = torch.LongTensor(random.sample(
                        range(src_emb.shape[0]), tgt_emb.shape[0])).to(self.device)
                    coral_loss = models.CORAL(torch.index_select(
                        src_emb, 0, random_index), tgt_emb, self.device)
                # Splitting the bucket to calculate CORAL Loss is not effective and is discarded
                # tgt_class_set = set(class_target.cpu().numpy().tolist())
                # coral_loss = 0
                # for tgt_label in tgt_class_set:
                #     index_src = torch.nonzero(class_train_batch==tgt_label).squeeze()
                #     index_tgt = torch.nonzero(class_target==tgt_label).squeeze()
                #     if index_src.shape[0] < index_tgt.shape[0]:
                #         index_tgt = torch.LongTensor(random.sample(index_tgt.cpu().numpy().tolist(), index_src.shape[0])).to(self.device)
                #     else:
                #         index_src = torch.LongTensor(random.sample(index_src.cpu().numpy().tolist(), index_tgt.shape[0])).to(self.device)
                #     coral_loss += models.CORAL(torch.index_select(src_emb, 0, index_src), torch.index_select(tgt_emb, 0, index_tgt), self.device)
                # Get the output of the C network
                outC = self.netC(src_emb)

                # Use the image generated by src to calculate the Loss with the real Label, in order for the features extracted by F to be better classified in D
                src_fakeoutputD_s, src_fakeoutputD_c, src_realoutputD_s_f, src_realoutputD_c_f = self.netD(
                    src_gen, src_emb)
                errF_src_fromG = (self.criterion_c(src_fakeoutputD_c, class_train_batch) +
                                  self.criterion_s(src_fakeoutputD_s, reallabel_s)) * (self.opt.alpha)
                errF_src_fromD = (self.criterion_c(src_realoutputD_c_f, class_train_batch) +
                                  self.criterion_s(src_realoutputD_s_f, reallabel_s)) * (self.opt.adv_weight)
                # Because F also needs to confuse src and tgt pictures, so here the output of tgt fake pictures and the loss of the real Label are calculated, in order to make the features output by F more easily discerned as true by D
                # The second Loss is used to enhance the tgt domain images generated by F and G for the identification of Class in D
                tgt_fakeoutputD_s, tgt_fakeoutputD_c, tgt_realoutputD_s_f_rvs, tgt_realoutputD_c_f = self.netD(
                    tgt_gen, tgt_emb, tgt_emb_rvs)
                errF_tgt_fromG = (self.criterion_s(tgt_fakeoutputD_s, reallabel_t) +
                                  self.criterion_c(tgt_fakeoutputD_c, class_target))*(self.opt.alpha)
                errF_tgt_fromD = (self.criterion_c(tgt_realoutputD_c_f, class_target) +
                                  self.criterion_s(tgt_realoutputD_s_f_rvs, fakelabel_t)) * (self.opt.adv_weight)

                # if epoch > 5000:
                # Get the Loss of C network
                errF_fromC = self.criterion_c(outC, class_train_batch)
                _lambda = (epoch / n_epochs) * 1e6
                # Loss_F = src image's classification Loss(From C) + src generated image's true/false classification and label classification Loss(From G) + tgt generated image's true/false classification and label classification Loss(From G) + src_F_emb true/false classification and label classification Loss(From D) + tgt_F_emb true/false classification and label classification Loss(From D) + CORAL Loss
                errF = errF_fromC + errF_src_fromG + errF_tgt_fromG + \
                    errF_src_fromD + errF_tgt_fromD + _lambda * coral_loss
                # else:
                # errF = errF_src_fromD + errF_tgt_fromD
                errF.backward()
                self.optimizerF.step()

                # Record F net train update
                writer.add_scalar('F_net/F_src_gen_loss',
                                  errF_src_fromG.detach().cpu().item(), curr_iter)
                writer.add_scalar('F_net/F_tgt_gen_loss',
                                  errF_tgt_fromG.detach().cpu().item(), curr_iter)
                writer.add_scalar('F_net/F_src_emb_loss',
                                  errF_src_fromD.detach().cpu().item(), curr_iter)
                writer.add_scalar('F_net/F_tgt_emb_loss',
                                  errF_tgt_fromD.detach().cpu().item(), curr_iter)
                writer.add_scalar(
                    'F_net/CORAL_loss', _lambda * coral_loss.detach().cpu().item(), curr_iter)
                writer.add_scalar('F_net/F_Loss_All',
                                  errF.detach().cpu().item(), curr_iter)

                # Updating end
                curr_iter += 1

                # Learning rate scheduling
                if self.opt.lrd:
                    self.optimizerD = utils.exp_lr_scheduler(
                        self.optimizerD, epoch, self.opt.lr, self.opt.lrd, curr_iter)
                    self.optimizerF = utils.exp_lr_scheduler(
                        self.optimizerF, epoch, self.opt.lr, self.opt.lrd, curr_iter)
                    self.optimizerC = utils.exp_lr_scheduler(
                        self.optimizerC, epoch, self.opt.lr, self.opt.lrd, curr_iter)
                    self.optimizerG = utils.exp_lr_scheduler(
                        self.optimizerG, epoch, self.opt.lr, self.opt.lrd, curr_iter)

            ######################
            # validate the model #
            ######################
            # Terminal logging
            ave_tgt_class_score = np.mean(tgt_class_scores)
            ave_tgt_class_loss = np.mean(tgt_class_losses)
            ave_c_score = np.mean(c_scores)
            ave_c_loss = np.mean(c_losses)
            epoch_len = len(str(n_epochs))

            if ave_tgt_class_score > best_tgt_score:
                best_tgt_score = ave_tgt_class_score
                best_tgt_report = tgt_report
            if ave_c_score > best_src_score:
                best_src_score = ave_c_score
                best_src_report = src_report

            if curr_iter % 50 == 0:
                print(f'[{epoch:>{epoch_len}}/{n_epochs:>{epoch_len}}] ' +
                      f'train_loss: {ave_c_loss:.5f} ' + f'train_score: {ave_c_score:.5f} ' +
                      f'tgt_class_loss: {ave_tgt_class_loss:.5f} ' + f'tgt_class_score: {ave_tgt_class_score:.5f} ' + f'CORAL_LOSS: {_lambda *coral_loss:.10f} ')

            # early_stopping needs the validation loss to check if it has decresed,
            # and if it has, it will make a checkpoint of the current model
            # if epoch > 5000:
            self.early_stopping(ave_tgt_class_loss, self.models)

            if self.early_stopping.early_stop:
                print("Early stopping")
                break

        # load the last checkpoint with the best model
        self.early_stopping.load_checkpoint(self.models)
        return best_src_score, best_src_report, best_tgt_score, best_tgt_report
