import torch
from client import *
from .server import Server
import torch.nn.functional as F
import copy
class FLOA(Server):
    def __init__(self, device, model_func, init_model, init_par_list, datasets, method, args):   
        super(FLOA, self).__init__(device, model_func, init_model, init_par_list, datasets, method, args)
        self.began = [0.5 for i in range(99)]
        self.a =  torch.zeros(init_par_list.shape[0])
        
        self.h_params_list = torch.zeros((args.total_client, init_par_list.shape[0]))
        print(" Dual Variable Param List  --->  {:d} * {:d}".format(
                self.clients_updated_params_list.shape[0], self.clients_updated_params_list.shape[1]))
        
        # rebuild
        self.comm_vecs = {
            'Params_list': init_par_list.clone().detach(),
            'Local_dual_correction': torch.zeros((init_par_list.shape[0])),
        }
        self.Client = floa
    
    
    def process_for_communication(self, client, Averaged_update):
        if not self.args.use_RI:
            self.comm_vecs['Params_list'].copy_(self.server_model_params_list)
        else:
            # RI adopts the w(i,t) = w(t) + beta[w(t) - w(i,K,t-1)] as initialization
            self.comm_vecs['Params_list'].copy_(self.server_model_params_list + self.args.beta\
                                    * (self.server_model_params_list - self.clients_params_list[client]))
        
        # self.comm_vecs['Local_dual_correction'].copy_(self.h_params_list[client] - self.server_model_params_list)
        self.comm_vecs['Local_dual_correction'].copy_(self.h_params_list[client] - self.comm_vecs['Params_list'])

    def global_update(self,selected_clients):
        mask = self.get_topk_pii_mask(kappa=0.3)



        cosine_sim= []
        h_params_pca_list_normal = []
        clients_params = self.clients_params_list[selected_clients]
        print(clients_params.shape)
        clients_params_root = clients_params[-1]

        for i in range(99):
            if i not in selected_clients:
                cosine_sim.append(0)
            if i in selected_clients:

                cosine_sim.append(F.cosine_similarity( torch.mul(mask[i], self.clients_params_list[i]), clients_params_root,dim=0))


        for i in range(99):
            client_params_list_normal = (torch.linalg.norm(clients_params_root) / torch.linalg.norm(
                self.clients_params_list[i])) * self.clients_params_list[i]
            h_params_pca_list_normal.append(client_params_list_normal)
        aa = copy.deepcopy(self.clients_params_list[-1])
        aa = aa  -  copy.deepcopy(self.clients_params_list[-1])
        su = 0
        for i in range(99):
            aa =  aa + h_params_pca_list_normal[i]*cosine_sim[i]
            su = su + cosine_sim[i]
        aa = aa/su
        a = aa + torch.mean(self.h_params_list, dim=0)
        return a

    def get_topk_pii_mask(self, kappa=0.3, eps=1e-8, exclude_root=True):
        """
        根据 PII 为每个客户端生成 mask：
        前 kappa 比例的最大 PII 参数位置为 1，其余为 0

        返回:
            mask: shape [num_clients, num_params]
        """
        pii = self.compute_pii(eps=eps, exclude_root=exclude_root)  # [K, D]

        K, D = pii.shape
        topk_num = max(1, int(D * kappa))  # 至少选1个

        # 对每个客户端分别取前 topk_num 个最大值的下标
        _, indices = torch.topk(pii, topk_num, dim=1)  # [K, topk_num]

        # 构造全0 mask
        mask = torch.zeros_like(pii, dtype=torch.float32)  # [K, D]

        # 将对应位置置为1
        mask.scatter_(1, indices, 1.0)

        return mask
    def compute_pii(self, eps=1e-8, exclude_root=True):
        """
        根据公式计算每个客户端每个参数的重要性 PII
        返回:
            pii: shape [num_clients, num_params]
        """

        # 如果 self.clients_params_list 是 list，先堆叠成 tensor
        if isinstance(self.clients_params_list, list):
            clients_params = torch.stack(self.clients_params_list, dim=0)
        else:
            clients_params = self.clients_params_list

        # 如果最后一个不是普通客户端，而是 root model update，需要排除
        if exclude_root:
            clients_params = clients_params[:-1]

        # clients_params: [K, D]
        abs_params = torch.abs(clients_params)  # [K, D]

        # 对每个参数维度 j，在所有客户端上取中位数
        med_params = torch.median(abs_params, dim=0).values  # [D]

        # 按公式计算
        pii = abs_params + torch.abs(abs_params - med_params) / (med_params + eps)  # [K, D]

        return pii

    
    def postprocess(self, client, received_vecs):
        self.h_params_list[client] += self.clients_updated_params_list[client]