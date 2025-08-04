import os
import os.path as osp
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import logging
import numpy as np
from model import SimVP, Discriminator
from tqdm import tqdm
from API import *
from utils import *
from torch.autograd import Variable
from PIL import Image
import imageio

import torchvision.models as models
import torchvision.transforms as transforms

cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

class PerceptualLoss(nn.Module):
    def __init__(self, layers=[0, 5], device=None):
        super(PerceptualLoss, self).__init__()
        #self.use_gpu = use_gpu
        vgg = models.vgg19(pretrained=True).features

        self.layers = layers
        self.vgg_layers = nn.ModuleList([vgg[i].eval() for i in layers])
        for param in self.vgg_layers.parameters():
            param.requires_grad = False

        self.vgg_layers = self.vgg_layers.to(device)

    def forward(self, x, y):
        loss = 0
        for i, layer in enumerate(self.vgg_layers):
            x = layer(x)
            y = layer(y)
            loss += nn.functional.mse_loss(x, y)
        return loss
    
def orthogonal_regularization(model, lambda_ortho=1e-5):
    ortho_loss = 0
    for name, param in model.named_parameters():
        if 'weight' in name and param.requires_grad:
            weight = param.view(param.size(0), -1)
            sym = torch.mm(weight, weight.t())
            sym -= torch.eye(weight.size(0)).to(weight.device)
            ortho_loss += lambda_ortho * sym.norm(2)
    return ortho_loss
    
class Exp:
    def __init__(self, args):
        super(Exp, self).__init__()
        self.args = args
        
        self.dataname = args.dataname
        
        self.ld = args.ld
        self.n_train = args.n_train
        self.n_test = args.n_test
        self.config = self.args.__dict__
        self.device = self._acquire_device()
        #self.device = torch.device('cuda:{}'.format(self.args.gpu))
        
        self.mse_step = args.mse_step
        self.gen_step = args.gen_step
        self.disc_step = args.disc_step
        
        self.lmbda_feat = args.lmbda_feat
        self.lmbda_orth = args.lmbda_orth
        
        self.config = f'{self.dataname}_trn{self.n_train}_tst{self.n_test}_lr{self.args.lr}_mse{self.mse_step}_gen{self.gen_step}_disc{self.disc_step}_feat{self.lmbda_feat}_orth{self.lmbda_orth}'

        self._preparation()
        print_log(output_namespace(self.args))

        self._get_data()
        self._select_optimizer()
        self._select_criterion()
        
        #self.perceptual_loss = PerceptualLoss(device=self.device).to(self.device)
        #self.transform = transforms.Compose([
        #    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        #])

    def _acquire_device(self):
        #if self.args.use_gpu:
        if True:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.args.gpu)
            device = torch.device('cuda:{}'.format(self.args.gpu))
            print_log('Use GPU: {}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            print_log('Use CPU')
        return device

    def _preparation(self):
        # seed
        set_seed(self.args.seed)
        # log and checkpoint
        self.path = osp.join(self.args.res_dir, self.args.ex_name)
        check_dir(self.path)

        self.checkpoints_path = osp.join(self.path, 'checkpoints')
        check_dir(self.checkpoints_path)

        sv_param = osp.join(self.path, 'model_param.json')
        with open(sv_param, 'w') as file_obj:
            json.dump(self.args.__dict__, file_obj)

        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        logging.basicConfig(level=logging.INFO, filename=osp.join(self.path, 'log.log'),
                            filemode='a', format='%(asctime)s - %(message)s')
        # prepare data
        self._get_data()
        # build the model
        self._build_model()

    def _build_model(self):
        args = self.args
        self.model = SimVP(tuple(args.in_shape), args.hid_S,
                           args.hid_T, args.N_S, args.N_T).to(self.device)
        self.disc = Discriminator().to(self.device)
        
        if self.args.train_step != 0:
            checkpoint = f'{self.checkpoints_path}/{self.args.train_step}.pth'
            self.model.load_state_dict(torch.load(checkpoint))
            print(f'Loaded {checkpoint}!')

    def _get_data(self):
        config = self.args.__dict__
        self.train_loader, self.vali_loader, self.test_loader, self.data_mean, self.data_std = load_data(**config)
        self.vali_loader = self.test_loader if self.vali_loader is None else self.vali_loader

    def _select_optimizer(self):
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.args.lr)
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer, max_lr=self.args.lr, steps_per_epoch=len(self.train_loader), epochs=self.args.epochs)
        
        self.optimizer_G = torch.optim.Adam(self.model.parameters(), lr=self.args.lr, betas=(0.5, 0.999))
        self.optimizer_D = torch.optim.Adam(self.disc.parameters(), lr=self.args.lr, betas=(0.5, 0.999))
        
        return self.optimizer

    def _select_criterion(self):
        self.criterion = torch.nn.MSELoss()
        self.criterion_GAN = torch.nn.MSELoss()
        self.criterion_pixelwise = torch.nn.L1Loss()

    def _save(self, name=''):
        torch.save(self.model.state_dict(), os.path.join(
            self.checkpoints_path, name + '.pth'))
        state = self.scheduler.state_dict()
        fw = open(os.path.join(self.checkpoints_path, name + '.pkl'), 'wb')
        pickle.dump(state, fw)


    def train(self, args):
        config = args.__dict__
        recorder = Recorder(verbose=True)
        train_step = self.args.train_step
        
        for epoch in range(self.args.init_epoch, config['epochs']):
            train_loss = []
            self.model.train()
            train_pbar = tqdm(self.train_loader)

            for batch_x, batch_y in train_pbar:
                train_step += 1

                #batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                _, _, _, W, H = batch_x.shape
                #print(batch_x.shape, batch_y.shape)

                real = batch_y.reshape(len(batch_y) * self.n_test, 1, W, H)
                
                step_choice = random.choices([0, 1, 2], weights=[self.mse_step, self.gen_step, self.disc_step])[0]

                self.optimizer_G.zero_grad()
                self.optimizer_D.zero_grad()
                
                fake = self.model(batch_x.to(self.device))
                fake = fake.reshape(len(fake) * self.n_test, 1, W, H)
                fake = torch.clamp(fake, min=0, max=1)
                    
                if step_choice == 0:
                    _fake = fake.reshape(len(batch_x), -1, 1, W, H)
                    _, fake_feat = self.disc(fake)
                    _, real_feat = self.disc(real.to(self.device))

                    loss_G_pixel = self.criterion_pixelwise(fake, real.to(self.device)) 
                    loss_G_feat = self.criterion_pixelwise(fake_feat, real_feat) 
                    loss_G_orth = orthogonal_regularization(self.model)
                    loss_G = loss_G_pixel + self.lmbda_feat * loss_G_feat + self.lmbda_orth * loss_G_orth 
                    print(loss_G_pixel.item(), loss_G_feat.item(), loss_G_orth.item())

                    loss_G.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer_G.step()

                    train_pbar.set_description('train loss:\tGenerator: {:.4f}'.format(loss_G.item()))

                else:
                    if step_choice == 1:
                        ### Generator ###
                        pred_fake, _ = self.disc(fake)
                        label_valid = Variable(torch.from_numpy(np.ones(pred_fake.shape, dtype=np.float32)), requires_grad=False).to(self.device)
                    
                        loss_G = self.criterion_GAN(pred_fake, label_valid)
                        loss_G_orth = orthogonal_regularization(self.model)
                        loss_G = loss_G + self.lmbda_orth * loss_G_orth

                        loss_G.backward()
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                        self.optimizer_G.step()

                        train_pbar.set_description('train loss:\tGenerator: {:.4f}'.format(loss_G.item()))

                    elif step_choice == 2:
                        ### Discriminator ###
                        pred_real, _ = self.disc(real.to(self.device))
                        pred_fake, _ = self.disc(fake.detach())
                        
                        label_fake = Variable(torch.from_numpy(np.zeros(pred_fake.shape, dtype=np.float32)), requires_grad=False).to(self.device)
                        label_valid = Variable(torch.from_numpy(np.ones(pred_fake.shape, dtype=np.float32)), requires_grad=False).to(self.device)
                        
                        # Real loss
                        loss_real = self.criterion_GAN(pred_real, label_valid)

                        # Fake loss
                        loss_fake = self.criterion_GAN(pred_fake, label_fake)

                        # Total loss
                        loss_D = 0.5 * (loss_real + loss_fake)

                        loss_D.backward()
                        torch.nn.utils.clip_grad_norm_(self.disc.parameters(), 1.0)
                        self.optimizer_D.step()

                        train_pbar.set_description('train loss:\tDiscriminator: {:.4f}'.format(loss_D.item()))



                with open(f'loss_{self.config}.txt', 'a') as f:
                    if step_choice == 0 or step_choice == 1:
                        f.write(f'{step_choice} {loss_G.item()}\n')
                    else:
                        f.write(f'{step_choice} {loss_D.item()}\n')

                if train_step % args.save_step == 0:
                    with torch.no_grad():
                        self._save(name=str(train_step))

                if train_step % 20 == 0:

                    real = real.reshape(-1, self.n_test, 1, W, H)
                    fake = fake.reshape(-1, self.n_test, 1, W, H)
                    real = real[:,-self.n_test:]
                    fake = fake[:,-self.n_test:]

                    real = real.reshape(-1, self.n_test, W, H)[0].detach().cpu().numpy()
                    fake = fake.reshape(-1, self.n_test, W, H)[0].detach().cpu().numpy()

                    real = np.clip(real * 255, 0, 255).astype(np.uint8)
                    fake = np.clip(fake * 255, 0, 255).astype(np.uint8)

                    for j in range(self.n_test):
                        Image.fromarray(real[j]).save(f'figures/ans_{self.config}_{j}.png')
                        Image.fromarray(fake[j]).save(f'figures/pred_{self.config}_{j}.png')

                    paths = [Image.open(f'figures/ans_{self.config}_{j}.png') for j in range(self.n_test)]
                    imageio.mimsave(f'videos/ans_{self.config}.gif', paths, duration=0.4)

                    paths = [Image.open(f'figures/pred_{self.config}_{j}.png') for j in range(self.n_test)]
                    imageio.mimsave(f'videos/pred_{self.config}.gif', paths, duration=0.4)


    def vali(self, vali_loader):
        self.model.eval()
        preds_lst, trues_lst, total_loss = [], [], []
        vali_pbar = tqdm(vali_loader)
        for i, (batch_x, batch_y) in enumerate(vali_pbar):
            if i * batch_x.shape[0] > 1000:
                break

            batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
            pred_y = self.model(batch_x)
            
            # For COMS dataset. For other datasets, remove the below line.
            pred_y = pred_y[:,:self.n_test,:,:,:]
            
            list(map(lambda data, lst: lst.append(data.detach().cpu().numpy()), [
                 pred_y, batch_y], [preds_lst, trues_lst]))
            
            loss = self.criterion(pred_y, batch_y)
            vali_pbar.set_description(
                'vali loss: {:.4f}'.format(loss.mean().item()))
            total_loss.append(loss.mean().item())

        total_loss = np.average(total_loss)
        preds = np.concatenate(preds_lst, axis=0)
        trues = np.concatenate(trues_lst, axis=0)
        mse, mae, ssim, psnr = metric(preds, trues, vali_loader.dataset.mean, vali_loader.dataset.std, True)
        print_log('vali mse:{:.4f}, mae:{:.4f}, ssim:{:.4f}, psnr:{:.4f}'.format(mse, mae, ssim, psnr))
        self.model.train()
        return total_loss

    def test(self, args):
        self.model.eval()
        inputs_lst, trues_lst, preds_lst = [], [], []
        for batch_x, batch_y in self.test_loader:
            pred_y = self.model(batch_x.to(self.device))
            
            # For COMS dataset. For other datasets, remove the below line.
            pred_y = pred_y[:,:self.n_test,:,:,:]
        
            list(map(lambda data, lst: lst.append(data.detach().cpu().numpy()), [
                 batch_x, batch_y, pred_y], [inputs_lst, trues_lst, preds_lst]))
        
        inputs, trues, preds = map(lambda data: np.concatenate(
            data, axis=0), [inputs_lst, trues_lst, preds_lst])

        folder_path = self.path+'/results/{}/sv/'.format(args.ex_name)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mse, mae, ssim, psnr = metric(preds, trues, self.test_loader.dataset.mean, self.test_loader.dataset.std, True)
        print_log('mse:{:.4f}, mae:{:.4f}, ssim:{:.4f}, psnr:{:.4f}'.format(mse, mae, ssim, psnr))

        for np_data in ['inputs', 'trues', 'preds']:
            np.save(osp.join(folder_path, np_data + '.npy'), vars()[np_data])
        return mse