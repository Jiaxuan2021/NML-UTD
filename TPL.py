import vit
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from collections import OrderedDict
import argparse
from utils import threshold
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from tqdm import tqdm
import random
import AE
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

class Vit_UTD(nn.Module):
    def __init__(self, num_bands, dim=32):
        super(Vit_UTD, self).__init__()

        self.detector = vit.ViT(
                                image_size = 1,
                                near_band = 1,
                                num_patches = num_bands,
                                num_classes = 1,
                                dim = 32,
                                depth = 4,
                                heads = 2,
                                mlp_dim = 32,
                                dropout = 0.1,
                                emb_dropout = 0.1,
                            )
        
        self.mlp_head = nn.Sequential(
                                nn.LayerNorm(dim),
                                nn.Linear(dim, 1),
                                nn.Sigmoid()
        )

    def forward(self, x):
        x1 = self.detector(x)
        x2 = self.mlp_head(x1)
        return x1, x2
    
class nonlinear_mapping(nn.Module):
    def __init__(self, num_bands, endmembers):
        super(nonlinear_mapping, self).__init__()
        self.nonlinear = nn.Sequential(
                                    nn.Linear(num_bands, 9*endmembers),
                                    nn.Sigmoid(),
                                    nn.Linear(9*endmembers, 18*endmembers),
                                    nn.Sigmoid(),
                                    nn.Linear(18*endmembers, num_bands),
                                    nn.Sigmoid(),
                                    nn.Linear(num_bands, num_bands)
        )

    def init_weights(self, path):
        self.load_state_dict(torch.load(path))

    def forward(self, x):
        x = self.nonlinear(x)
        return x
    
def amend_Triplet_loss(n, p, a, margin1=0.6, margin2=0.8, epsilon=0.05):
    cos_pn = nn.functional.cosine_similarity(p, n).abs()
    cos_pa = nn.functional.cosine_similarity(p, a).abs()
    cos_an = nn.functional.cosine_similarity(a, n).abs()
    triplet_loss_1 = torch.mean(torch.max(cos_pn, cos_an) - cos_pa + margin2)

    ap_loss = F.relu(cos_pa - margin1 + epsilon)
    an_loss = F.relu(margin2 - cos_an)
    pn_loss = F.relu(margin2 - cos_pn)
    triplet_loss_2 = (ap_loss + an_loss + pn_loss).mean()
    return triplet_loss_1 + triplet_loss_2
    
def train(args, seed, data_hsi):
    seed_torch(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  
    data = data_hsi['data']
    prior = data_hsi['target']
    gt = data_hsi['gt']
    x_dim, y_dim, num_bands = data.shape
    hsi = data.reshape(-1, num_bands)
    validation_dataset = AE.UnderwaterDataset(hsi)
    validation_loader = DataLoader(validation_dataset, batch_size=args.batch_size, shuffle=False)

    mask = np.load(f'water_mask/NDWI_{args.data_name}.npy')
    water_sample = data[np.where(mask == 0)]
    land_sample = data[np.where(mask == 255)]
    if len(water_sample) < len(land_sample):
        repeat_times = len(land_sample) // len(water_sample) + 1
        water_sample = np.tile(water_sample, (repeat_times, 1))[:len(land_sample)]
    elif len(water_sample) > len(land_sample):
        repeat_times = len(water_sample) // len(land_sample) + 1
        land_sample = np.tile(land_sample, (repeat_times, 1))[:len(water_sample)]
    pseudo_data_num = max(water_sample.shape[0], land_sample.shape[0])
    water_dataset = AE.UnderwaterDataset(water_sample)
    land_dataset = AE.UnderwaterDataset(land_sample)
    water_loader = DataLoader(water_dataset, batch_size=args.batch_size, shuffle=True)
    land_loader = DataLoader(land_dataset, batch_size=args.batch_size, shuffle=True)
    pseudo_data_loader = DataLoader(water_dataset, batch_size=args.batch_size, shuffle=True)    # synthetic pseudo data while training

    model = Vit_UTD(num_bands).to(device)
    nonlinear = nonlinear_mapping(num_bands, args.endmembers)
    nonlinear.init_weights(args.nonlinear_init_weight)
    nonlinear = nonlinear.to(device).float()
    optimizer = torch.optim.Adam([
                                {'params': model.parameters()},
                                {'params': nonlinear.parameters(), 'lr': args.lr*0.01, 'weight_decay': 0}], lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.BCEWithLogitsLoss()
    Loss = []
    best_auc = 0

    for epoch in tqdm(range(1, args.epochs+1)):
        model.train()
        nonlinear.train()
        epoch_loss = 0
        for i, (water, land, pseudo_data) in enumerate(zip(water_loader, land_loader, pseudo_data_loader)):
            water = water.to(device).float()
            land = land.to(device).float()
            pseudo_data = pseudo_data.to(device)
            prior_batch = torch.tensor(np.tile(prior, (pseudo_data.shape[0], 1))).to(device)
            pseudo_abundance = torch.tensor(np.tile(np.random.rand(pseudo_data.shape[0])*0.1+0.9, (args.num_bands, 1)).T).to(device)
            pseudo_data = pseudo_data * (1 - pseudo_abundance) + prior_batch * pseudo_abundance
            pseudo_data = pseudo_data.float()

            optimizer.zero_grad()
            pseudo_target_sample = nonlinear(pseudo_data) + pseudo_data   # postnonlinear mixing model
            water_embedding, water_output = model(water)
            land_embedding, land_output = model(land)
            pseudo_embedding, pseudo_output = model(pseudo_target_sample)
            land_loss = criterion(land_output, torch.zeros_like(land_output))
            pseudo_loss = criterion(pseudo_output, torch.ones_like(pseudo_output) - 0.01)   # soft label
            loss_cls = land_loss + pseudo_loss*1.5
            loss_triplet = amend_Triplet_loss(land_embedding, water_embedding, pseudo_embedding)
            loss = loss_cls + loss_triplet
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                epoch_loss += loss.item()
        with torch.no_grad():
            epoch_loss /= len(water_loader)
            Loss.append(epoch_loss)
            # validation
            model.eval()
            nonlinear.eval()
            result = torch.zeros(x_dim*y_dim).to(device)
            batch_index = 0
            for batch in validation_loader:
                batch = batch.to(device).float()
                _, output = model(batch)
                start_index = batch_index * args.batch_size
                end_index = start_index + batch.shape[0]
                result[start_index:end_index] = output.squeeze()
                batch_index += 1
            result = result.cpu().numpy().reshape(x_dim, y_dim)
            if not os.path.exists(f'{args.save_path}/{args.data_name}'):
                os.makedirs(fr'{args.save_path}/{args.data_name}')

            epoch_auc = AE.get_epoch_AUC(result, args.data_name, gt)
            if epoch > 1:    # abandon the first epoch result
                if epoch_auc > best_auc:
                    best_auc = epoch_auc
                    torch.save(model.state_dict(), f'{args.save_path}/{args.data_name}/best_{args.data_name}_{seed}.pt')
        tqdm.write(fr'Epoch {epoch}/{args.epochs} Loss: {epoch_loss:.4f} AUC: {epoch_auc:.4f} Best AUC: {best_auc:.4f}')
    plt.plot(range(1, args.epochs+1), Loss, c='b')
    plt.title('Loss Curve')
    plt.savefig(fr'{args.save_path}/{args.data_name}/loss_{seed}.png')
    plt.clf()
    plt.close()

def test(args, seed, data_hsi):
    seed_torch(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  
    data = data_hsi['data']
    gt = data_hsi['gt']
    x_dim, y_dim, num_bands = data.shape
    hsi = data.reshape(-1, num_bands)
    model = Vit_UTD(num_bands).to(device)
    test_dataset = AE.UnderwaterDataset(hsi)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    model.load_state_dict(torch.load(f'{args.save_path}/{args.data_name}/best_{args.data_name}_{seed}.pt'))
    model.eval()
    with torch.no_grad():
        result = torch.zeros(x_dim*y_dim).to(device)
        batch_index = 0
        for batch in test_loader:
            batch = batch.to(device).float()
            _, output = model(batch)
            start_index = batch_index * args.batch_size
            end_index = start_index + batch.shape[0]
            result[start_index:end_index] = output.squeeze()
            batch_index += 1
        result = result.cpu().numpy().reshape(x_dim, y_dim)
        
        fpr, tpr, _ = roc_curve(gt.reshape(-1), result.reshape(-1))
        auc_roc = auc(fpr, tpr)
        plt.title('AUC={:.4f}'.format(auc_roc))
        plt.xlabel('FPR', fontsize=8)
        plt.ylabel('TPR', fontsize=8)
        plt.xlim([0.0,1.0])
        plt.ylim([0.0,1.0])
        plt.plot(fpr,tpr,color='b',linewidth=3)
        plt.plot([0, 1], [0, 1], color='m', linestyle='--')
        plt.savefig(f'{args.save_path}/{args.data_name}/ROC_{seed}.png')
        plt.clf()
        plt.close()
        result_threshold = threshold.threshold(result, threshold=0.1)
        np.save(f'{args.save_path}/{args.data_name}/result_{seed}.npy', result_threshold)
        plt.imshow(result_threshold)
        plt.axis('off')
        plt.savefig(f'{args.save_path}/{args.data_name}/detection_map_{seed}_result.png', bbox_inches='tight', pad_inches=0)
        plt.clf()
        plt.close()

def get_args():
    parser = argparse.ArgumentParser(description='Triplet Pseudo Data Learning')
    parser.add_argument('--data_name', type=str, default='ningxiang', help='The name of the dataset')
    parser.add_argument('--num_bands', type=int, default=273, help='The number of bands')
    parser.add_argument('--endmembers', type=int, default=5, help='The number of endmembers')
    parser.add_argument('--epochs', type=int, default=100, help='The number of epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='The batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='The learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='The weight decay')
    parser.add_argument('--device', type=str, default='0', help='The device to use')
    parser.add_argument('--save_path', type=str, default='result_detection', help='The path to save the result')
    parser.add_argument('--nonlinear_init_weight', type=str, default='result_AE/ningxiang/nonlinear_AE.pt', help='The path to the initial weight of nonlinear mapping in stage 1')
    return parser.parse_args()

def main(seed):
    args = get_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device 
    data_hsi = sio.loadmat(f'dataset/{args.data_name}/data.mat')
    print(args)
    train(args, seed, data_hsi)
    print('Finish training!, start testing...')
    test(args, seed, data_hsi)

seed_list = [41, 42, 43, 44, 45, 46, 47, 48, 49]

if __name__ == '__main__':
    for idx, seed in enumerate(seed_list):
        print("[%d / %d Random seed (%d) start training]" %(idx+1, len(seed_list), seed_list[idx]))
        main(seed)