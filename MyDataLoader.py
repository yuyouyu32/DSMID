from sklearn.model_selection import KFold
import torch
import os

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")


class MyDataLoader:
    """This class defines the data loading methods.

    Attributes:
        data_path: The file path of data.
        source_img: The file path of source domain img.
        target_img: The file path of target domain img.
        targets: The targets' names of job.

    """

    def __init__(self, source_img: str, target_img: str, orginal_data: str, target_data, targets: str, t_target) -> None:
        # if os.path.isfile(source_img) and os.path.isfile(target_img) and os.path.isfile(orginal_data) and os.path.isfile(target_data):
        self.source_img = source_img
        self.target_img = target_img
        self.orginal_data = orginal_data
        self.target_data = target_data

        # else:
        #     raise Exception(
        #         "Sorry, the path of data or path of images is illegal.")
        if len(targets) <= 0 or len(t_target) <= 0:
            raise Exception(
                "The names of targets can not be empty.")
        self.targets = targets
        self.t_target = t_target

    def set_source_img(self, source_img: str) -> None:
        """Set new path of images.

        Arguments:
            source_img: The direction path of images.

        """
        if os.path.isfile(source_img):
            self.source_img = source_img
        else:
            raise Exception("Sorry, the path of saving is illegal.")

    def set_target_img(self, target_img: str) -> None:
        """Set new path of images.

        Arguments:
            target_img: The direction path of images.

        """
        if os.path.isfile(target_img):
            self.target_img = target_img
        else:
            raise Exception("Sorry, the path of saving is illegal.")

    def set_orginal_data(self, original_data: str) -> None:
        """Set new path of original data.

        Arguments:
            orginal_data: The direction path of images.

        """
        if os.path.isfile(original_data):
            self.orginal_data
        else:
            raise Exception("Sorry, the path of saving is illegal.")

    def set_targets(self, targets: str) -> None:
        """Set new names of targets.

        Arguments:
            targets: The targets' names of job.

        """
        if len(targets) <= 0:
            raise Exception(
                "The names of targets can not be empty.")
        self.targets = targets

    def get_dataset(self, if_norm=False, get_scaler=False):
        # source domain
        source_domain = pd.read_csv(self.source_img)
        original_data = pd.read_csv(self.orginal_data)

        # traget domain
        target_domain = pd.read_csv(self.target_img)
        target_domain_x = target_domain.astype('double').values
        target_data = pd.read_csv(self.target_data)
        target_domain_class = target_data.loc[:, 'class'].values
        target_domain_y = target_data.loc[:, self.t_target].values

        n_classes = len(set(target_domain_class))
        # Delete class which does not appear in target domain.
        indexes = original_data[original_data['class'].isin(
            set(target_domain_class))].index
        original_data = original_data.iloc[indexes, :]
        source_domain = source_domain.iloc[indexes, :]

        source_domain_y = original_data.loc[:, self.targets].values
        source_domain_class = original_data.loc[:, 'class'].values
        source_domain_x = source_domain.astype('double').values

        # normolization
        if if_norm:
            len_source = len(source_domain_x)
            len_target = len(target_domain_x)
            all_x = np.concatenate((source_domain_x, target_domain_x), axis=0)
            from sklearn.preprocessing import MinMaxScaler, StandardScaler
            scaler = StandardScaler()
            all_x = scaler.fit_transform(all_x)
            source_domain_x, target_domain_x = all_x[:len_source,
                                                     :], all_x[len_source:, :]
            assert len(source_domain_x) == len_source
            assert len(target_domain_x) == len_target

        # source domain
        source_domain_x = torch.from_numpy(
            source_domain_x).reshape(-1, 1, 24, 21)
        source_domain_y = torch.from_numpy(source_domain_y)
        source_domain_class = torch.from_numpy(source_domain_class).long()

        target_domain_x = torch.from_numpy(
            target_domain_x).reshape(-1, 1, 24, 21)
        target_domain_y = torch.from_numpy(target_domain_y)
        target_domain_class = torch.from_numpy(target_domain_class).long()

        if if_norm and get_scaler:
            return source_domain_x, source_domain_y, source_domain_class, target_domain_x, target_domain_y, target_domain_class, original_data, n_classes, scaler
        else:
            return source_domain_x, source_domain_y, source_domain_class, target_domain_x, target_domain_y, target_domain_class, original_data, n_classes


if __name__ == '__main__':
    dataloader = MyDataLoader(source_img='../Data/Schedule_HV.csv', orginal_data='../Data/HV_O_data.csv', targets='Hardness (HV)',
                              target_img='../Data/Schedule_UTS.csv', target_data='../Data/UTS_O_data.csv', t_target='UTS (MPa)')
    source_domain_x, source_domain_y, source_domain_class, target_domain_x, target_domain_y, target_domain_class = dataloader.get_dataset()
    print(source_domain_x.shape, source_domain_y.shape, source_domain_class.shape)
    print(target_domain_x.shape, target_domain_y.shape, target_domain_class.shape)
