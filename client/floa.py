import torch
from .client import Client
from utils import *
from optimizer import *


class floa(Client):
    def __init__(self, device, model_func, received_vecs, dataset, lr, args):   
        super(floa, self).__init__(device, model_func, received_vecs, dataset, lr, args)
        
        # rebuild
        self.base_optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, weight_decay=self.args.weight_decay+self.args.lamb)
        self.optimizer = ESAM(self.model.parameters(), self.base_optimizer, rho=self.args.rho)
    
    
    def train(self):
        # local training
        self.model.train()
        
        for k in range(self.args.local_epochs):
            for i, (inputs, labels) in enumerate(self.dataset):
                inputs = inputs.to(self.device)
                labels = labels.to(self.device).reshape(-1).long()
                
                self.optimizer.paras = [inputs, labels, self.loss, self.model]
                self.optimizer.step()
                
                param_list = param_to_vector(self.model)
                delta_list = self.received_vecs['Local_dual_correction'].to(self.device)
                loss_correct = self.args.lamb * torch.sum(param_list * delta_list)
                
                loss_correct.backward()
                
                # Clip gradients to prevent exploding
                torch.nn.utils.clip_grad_norm_(parameters=self.model.parameters(), max_norm=self.max_norm)
                self.base_optimizer.step()    
                
        last_state_params_list = get_mdl_params(self.model)
        self.comm_vecs['local_update_list'] = last_state_params_list - self.received_vecs['Params_list']
        self.comm_vecs['local_model_param_list'] = last_state_params_list

        return self.comm_vecs
    def train1(self):
        # local training
        self.model.train()
        for k in range(self.args.local_epochs):
            for i, (inputs, labels) in enumerate(self.dataset):
                inputs = inputs.to(self.device)
                labels = labels.to(self.device).reshape(-1).long()

                self.optimizer.paras = [inputs, labels, self.loss, self.model]
                self.optimizer.step()
                # print(self.received_vecs, "1233")
                param_list = param_to_vector(self.model)
                # print(param_list.shape,"311")
                delta_list = self.received_vecs['Local_dual_correction'].to(self.device)
                # print(delta_list.shape, "312")
                # print(param_list * delta_list)
                # print(self.args.lamb,"313")
                a = param_list * delta_list
                # print(a.shape,"315")
                loss_correct = self.args.lamb * torch.sum(param_list * delta_list)
                # print(loss_correct,"314")
                loss_correct.backward()

                # Clip gradients to prevent exploding
                torch.nn.utils.clip_grad_norm_(parameters=self.model.parameters(), max_norm=self.max_norm)
                self.base_optimizer.step()
        # print("1111111111111111111111111111111111")
        # for parameters in self.model.parameters():
        #     print(parameters)
        # print("1111111111111111111111111111111111")
        # for parameters in self.model.parameters():
        #     print(parameters)
        for name, param in self.model.named_parameters():
            if 'bias' in name or 'bn' in name:
                # 不对偏置和BatchNorm的参数添加噪声
                continue
            noise = torch.normal(0, 0.01, size=param.size()).cuda()
            param.data.add_(noise)

        last_state_params_list = get_mdl_params(self.model)
        self.comm_vecs['local_update_list'] = last_state_params_list - self.received_vecs['Params_list']
        self.comm_vecs['local_model_param_list'] = last_state_params_list

        return self.comm_vecs