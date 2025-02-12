import torch
from torch.utils.data import DataLoader
from torch import nn
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from typing import Callable, Optional

AVAIL_GPUS = min(1, torch.cuda.device_count())
#BATCH_SIZE = 256 if AVAIL_GPUS else 64

BATCH_SIZE = 1000 if AVAIL_GPUS else 64
class TabularDataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        super(TabularDataset, self).__init__()
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        #self.dist = torch.FloatTensor(dist)

    def __len__(self):
        #return self.sheaf_data.size(0)
        # HACKY SOLUTION
        return self.y.shape[0]

    def __getitem__(self, index):
        return self.X[index], self.y[index]


class TabularDataModule(LightningDataModule):
    def __init__(self, file_dict, batch_size :int = BATCH_SIZE):
        super().__init__()
        self.file_dict = file_dict
        self.batch_size = batch_size

    # When doing distributed training, Datamodules have two optional arguments for
    # granular control over download/prepare/splitting data:

    # OPTIONAL, called only on 1 GPU/machine
    #def prepare_data(self):
        #MNIST(os.getcwd(), train=True, download=True)
        #MNIST(os.getcwd(), train=False, download=True)


    # OPTIONAL, called for every GPU/machine (assigning state is OK)
    def setup(self, stage: Optional[str] = None):
        # transforms
        # split dataset
        if stage in (None, "fit"):
            #self.tabular_train = torch.load(os.path.join(self.file_dict['train']))
            #self.tabular_val   = torch.load(os.path.join(self.file_dict['val']))
            self.tabular_train = self.file_dict['train']
            self.tabular_val   = self.file_dict['val']

        if stage == "test":
            #self.tabular_test = torch.load(os.path.join(self.file_dict['test']))
            self.tabular_test = self.file_dict['test']
        if stage == "predict":
            #self.tabular_predict = torch.load(os.path.join(self.file_dict['predict']))
            self.tabular_predict = self.file_dict['predict']
        # return the dataloader for each split
    def train_dataloader(self):
        tabular_train = DataLoader(self.tabular_train, batch_size=self.batch_size, shuffle=True, num_workers=4)
        return tabular_train

    def val_dataloader(self):
        tabular_val = DataLoader(self.tabular_val, batch_size=self.batch_size, num_workers=4)
        return tabular_val

    def test_dataloader(self):
        tabular_test = DataLoader(self.tabular_test, batch_size=self.batch_size)
        return tabular_test

    def predict_dataloader(self):
        tabular_predict = DataLoader(self.tabular_predict, batch_size=self.batch_size)
        return tabular_predict


class MLP(nn.Module):
    """docstring for MLP"""
    def __init__(self, input_dim, embed_dim, layer_widths):
        super(MLP, self).__init__()
        self.input_dim = input_dim
        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat,out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(num_features=out_feat))
            layers.append(nn.LeakyReLU())
            layers.append(nn.Dropout(p=0.1, inplace = False))
            return layers

        self.fcblock = nn.Sequential(
                                     *block(input_dim, layer_widths[0] ),
                                     *[layers for i in range(len(layer_widths)-1) for layers in block(layer_widths[i],layer_widths[i+1])],
                                     nn.Linear(layer_widths[-1], embed_dim)
                                     )
    def forward(self, src):
        output = src.reshape(-1,self.input_dim)
        output = self.fcblock(output)
        return output


class Classifier(LightningModule):
    """docstring for Classifier"""
    def __init__(self, data_type, backbone_type, learning_rate, modelparams):
        super(Classifier, self).__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.activation = nn.Sigmoid()
        self.loss = nn.BCELoss()

        if data_type == 'tabular':
            if backbone_type == 'MLP':
                self.layers = MLP(*modelparams)


            else:
                raise LookupError('only supports MLP')

        else:
            raise LookupError('Only supports tabular data for now')


    def forward(self, x):

        output = self.layers(x)
        return output


    def training_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forwardset
        loss = self._common_step(batch, batch_idx, "train")
        return loss

    def validation_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward
        loss = self._common_step(batch, batch_idx, "val")
        return loss
    
    def test_step(self, batch, batch_idx):
        if batch_idx == 0:
            print('start')

        self._common_step(batch, batch_idx, "test")
        
    
    def _common_step(self, batch, batch_idx, stage: str):



        x, y = batch
        #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")        
        loss = self.loss(self.activation(torch.squeeze(self(x))), y)

        if stage != 'test':
            self.log(f"{stage}_loss", loss, prog_bar=True, on_step=True)
        return loss



    def predict_step(self, batch, batch_idx: int, dataloader_idx: int = None):
        x, label = batch
        return self(x), label

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.layers.parameters(), lr=self.learning_rate)
        return optimizer