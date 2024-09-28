import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from collections import OrderedDict
import argparse
from utils import train_objectives, generate_init_weight, threshold
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from tqdm import tqdm
import random
import scipy.io as sio


def seed_torch(seed): 
    '''
    Keep the seed fixed thus the results can keep stable
    '''
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

class GaussianDroupout(nn.Module): 
    """
    Multiplied by a Gaussian sequence, 
    each forward propagation will be slightly different, 
    introducing a certain randomness
    """
    def __init__(self, alpha=1.0): # alpha is the variance of Gaussian distribution
        super(GaussianDroupout, self).__init__()
        self.alpha = alpha

    def forward(self, x):
        """
        N(1, alpha)
        """
        if self.training:
            # Gaussian noise
            noise = torch.randn(x.size()) * self.alpha + 1   # mean=1, std=self.alpha
            if x.is_cuda:
                noise = noise.cuda()
                return x * noise
            else:
                raise RuntimeError('GaussianDroupout is only supported on CUDA while training')
        else:
            return x
        
class ASC(nn.Module):
    """
    Abundances sum to one and constraint
    """
    def __init__ (self):
        super(ASC, self).__init__()
    
    def forward(self, input):
        # ANC ASC
        constrained = F.softmax(input, dim=1)

        return constrained
    
class UnderwaterDataset(Dataset):
    """
    Input data: N x WAVELENGTHS
    """
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        data = self.data[index]
        data = torch.tensor(data)
        return data
    
    def __len__(self):
        return self.data.shape[0]
    
def get_epoch_AUC(result_map, data_name, gt):
    """
    Calculate the AUC of the result map
    """
    min_val = np.min(result_map)
    max_val = np.max(result_map)
    range_val = max_val - min_val
    if range_val == 0:
        print("Warning: min and max values are equal. Division by zero.")
        print(min_val, max_val)   
        return 0
    else:
        result_norm = (result_map - min_val) / range_val
    gt = gt.flatten()
    result_norm = result_norm.flatten()
    FPR, TPR, _ = roc_curve(gt, result_norm)
    AUC = auc(FPR, TPR)
    return AUC
    
class AutoEncoder(nn.Module):
    """
    Autoencoder for nonlinear part to learn the nonlinear mapping of different water environments
    Too deep neural network may cause overfitting 
    """
    def __init__(self, data_name, num_bands, endmembers, activation: str='LeakyReLU'):
        super(AutoEncoder, self).__init__()
        self.data_name = data_name
        self.num_bands = num_bands
        self.endmembers = endmembers
        self.activation = getattr(nn, activation)()

        self.asc = ASC()
        self.gauss = GaussianDroupout()

        self.encoder = nn.Sequential(
                                    nn.Linear(num_bands, 18*endmembers),
                                    self.activation,
                                    nn.Linear(18*endmembers, 9*endmembers),
                                    self.activation,
                                    nn.Linear(9*endmembers, 6*endmembers),
                                    self.activation,
                                    nn.Linear(6*endmembers, 3*endmembers),
                                    self.activation,
                                    nn.Linear(3*endmembers, endmembers+1),
                                    self.activation,
                                    nn.BatchNorm1d(endmembers+1),
                                    nn.Softplus(threshold=5),
                                    self.asc
        )

        self.middle = self.gauss

        self.decoder_linear = nn.Sequential(OrderedDict([
                                                ('Linear1', nn.Linear(endmembers+1, num_bands, bias=False)),                    
        ]))

        self.nonlinear = nn.Sequential(
                                    nn.Linear(num_bands, 9*endmembers),
                                    nn.Sigmoid(),
                                    nn.Linear(9*endmembers, 18*endmembers),
                                    nn.Sigmoid(),
                                    nn.Linear(18*endmembers, num_bands),
                                    nn.Sigmoid(),
                                    nn.Linear(num_bands, num_bands)
        )
    
    def init_decoder_linear(self):
        """"
        Initialize the decoder linear layer with water spectrum and target prior spectrum
        B WAVELENGTHS
        weight: B x 2
        """
        if not os.path.exists(fr'init_weight/{self.data_name}.npy'):
            print("The initial weight does not exist, generating the initial weight...")
            generate_init_weight.generate_init_weight(self.data_name)
        init_weight = np.load(fr'init_weight/{self.data_name}.npy')
        self.decoder_linear.Linear1.weight = nn.Parameter(torch.from_numpy(init_weight).float())

    def forward(self, x):
        x = self.encoder(x)
        x1 = self.middle(x)
        x1 = self.decoder_linear(x1)
        x2 = self.nonlinear(x1) + x1   # postnonlinear model
        return x, x2
        
def train(args, data, seed):
    seed_torch(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x_dims, y_dims, num_bands = data.shape
    hsi = torch.tensor(data.reshape(-1, num_bands)).to('cuda')   # for validation

    dataset = UnderwaterDataset(data.reshape(-1, num_bands))
    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    model = AutoEncoder(args.data_name, args.num_bands, args.endmembers, args.activation)
    model.init_decoder_linear()
    decoder_linear_params = list(map(id, model.decoder_linear.parameters()))  
    other_params = filter(lambda p: id(p) not in decoder_linear_params, model.parameters())  

    optimizer = torch.optim.Adam([
                                {'params': other_params}, 
                                {'params': model.decoder_linear.parameters(), 'lr': args.lr*0.01, 'weight_decay': 0}   # fine-tune
    ], lr=args.lr, weight_decay=args.weight_decay) 

    model.to(device)
    Loss = []
    best_auc = 0
    for epoch in tqdm(range(1, args.epochs+1)):
        model.train()
        iterator = iter(train_loader)
        epoch_loss = 0
        for i in range(len(iterator)):
            batch = next(iterator)
            batch = batch.to(device)
            optimizer.zero_grad()
            enc_out, dec_out = model(batch.float())
            reconstruction_loss = 80 * nn.MSELoss()(dec_out, batch.float())
            abundance_regularization = 1e-4 * torch.log10(1 / (torch.var(enc_out, dim=1, unbiased=False) + 1e-8)).sum()
            loss = reconstruction_loss + abundance_regularization
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            epoch_loss += loss.item()
            epoch_loss /= len(iterator)
            Loss.append(epoch_loss)
        
        model.eval()
        if not os.path.exists(fr'{args.save_path}/{args.data_name}'):
            os.makedirs(fr'{args.save_path}/{args.data_name}')

        tqdm.write('Epoch [{}/{}], G_Loss: {:.8f}'.format(epoch, args.epochs, epoch_loss))
    nonlinear_state_dict = {k: v for k, v in model.state_dict().items() if 'nonlinear' in k}
    torch.save(nonlinear_state_dict, '{}/{}/{}_AE.pt'.format(args.save_path, args.data_name, args.save_name))
    torch.save(model.state_dict(), f'{args.save_path}/{args.data_name}/AE_stage1.pt')

    plt.plot(range(1, args.epochs+1), Loss, c='b', label='Loss')
    plt.legend()
    plt.xlabel('Epoch')
    plt.title('Loss Curve')
    plt.savefig(f'{args.save_path}/{args.data_name}/loss_curve.png')
    plt.clf()
    plt.close()

def test(args, data, gt, seed):
    seed_torch(seed)
    hsi = torch.tensor(data.reshape(-1, args.num_bands)).to('cuda')
    model = AutoEncoder(args.data_name, args.num_bands, args.endmembers, args.activation)
    model.load_state_dict(torch.load(f'{args.save_path}/{args.data_name}/AE_stage1.pt'))
    model.to('cuda')
    model.eval()
    with torch.no_grad():
        enc_out, _ = model(hsi)
        detect_result = enc_out.detach().cpu().squeeze().numpy().T[-1].reshape(data.shape[0], data.shape[1])
        auc = get_epoch_AUC(detect_result, args.data_name, gt)    # result in first stage
        np.save(f'{args.save_path}/{args.data_name}/ae_result_{auc:.4f}.npy', detect_result)
        plt.imshow(detect_result)
        plt.title(fr'AUC: {auc:.4f}')
        plt.axis('off')
        plt.savefig(f'{args.save_path}/{args.data_name}/ae_result_{auc:.4f}.png')
        plt.clf()
        plt.close()

        after_threshold = threshold.threshold(detect_result, 0.1)
        plt.imshow(after_threshold)
        plt.axis('off')
        plt.savefig(f'{args.save_path}/{args.data_name}/threshold.png', bbox_inches='tight', pad_inches=0)
        plt.clf()
        plt.close()

def get_args():
    parser = argparse.ArgumentParser(description='Autoencoder for training the nonlinear mapping of different water environments')
    parser.add_argument('--data_name', type=str, default='ningxiang', help='The name of the dataset')
    parser.add_argument('--num_bands', type=int, default=273, help='The number of bands')
    parser.add_argument('--endmembers', type=int, default=5, help='Total number of endmembers')
    parser.add_argument('--epochs', type=int, default=200, help='The number of epochs')
    parser.add_argument('--batch_size', type=int, default=512, help='The batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--save_path', type=str, default='result_AE', help='The path to save the temp result')
    parser.add_argument('--save_name', type=str, default='nonlinear', help='The name to save the nonlinear model weights')
    parser.add_argument('--activation', type=str, default='LeakyReLU', help='The activation function')
    parser.add_argument('--device', type=str, default='0', help='The device to run the model')
    return parser.parse_args()

def main(seed):
    args = get_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device 
    dataset = sio.loadmat(fr'dataset/{args.data_name}/data.mat')
    data = dataset['data']
    gt = dataset['gt']
    print(args)  
    train(args, data, seed)
    print('Training finished! Start testing...')
    test(args, data, gt, seed)

seed_list = [44]                     

if __name__ == '__main__':
    for seed in seed_list:
        main(seed)