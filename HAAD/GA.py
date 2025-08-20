import numpy as np
import pygad
import torch
import torch.nn as nn
import numpy as np
import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__),'../'))
sys.path.append(parent_dir)
from utils.data import *
from Ant_algorithm import *
from DLWF_pytorch.model import *
from tqdm import tqdm
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
print("GA.py working directory:", current_dir)
class universal_solution():
    def __init__(self, X, patches, model, train_loader):
        self.X = X
        self.patches = patches  # 要选多少个
        self.model = model
        self.train_loader = train_loader

    def evaluate_defense(self, selected_patches):
        # pdb.set_trace()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        sum_acc = 0
        for batch_x, batch_y in self.train_loader:
            batch_x_tensor = batch_x.float().to(device)
            batch_y_tensor = batch_y.float().to(device)
            selected_patches = torch.tensor(selected_patches).detach().to(device)
            adv_trace = generate_adv_trace(selected_patches, batch_x_tensor)
            
            predict = self.model(adv_trace)
            sum_acc += (predict.argmax(-1) == batch_y_tensor.argmax(-1)).sum().float()
        print("sum_acc:",sum_acc)
        return (len(self.train_loader.dataset)-sum_acc).item()# 取标量

    def fitness_func(self,  ga_instance, solution, solution_idx):

        binary_solution = np.zeros_like(solution)
        topk_indices = np.argsort(-solution)[:self.patches]
        binary_solution[topk_indices] = 1

        selected_patches = [self.X[i] for i in np.where(binary_solution == 1)[0]]
        reward = self.evaluate_defense(selected_patches)
        return reward  # 遗传算法默认是 maximization

    def run_optimization(self):
        # pdb.set_trace()
        num_genes = len(self.X)

        ga_instance = pygad.GA(
            num_generations=50,
            num_parents_mating=10,
            fitness_func=self.fitness_func,
            sol_per_pop=10,
            num_genes=num_genes,
            gene_type=float,
            init_range_low=0.0,
            init_range_high=1.0,
            mutation_type="random",  # 默认就是这个
            # mutation_by=0.3,         # 控制最大变异幅度
            mutation_percent_genes=10,
        )

        ga_instance.run()
        best_solution, best_fitness, _ = ga_instance.best_solution()
        selected = np.zeros_like(best_solution)
        topk = np.argsort(-best_solution)[:self.patches]
        selected[topk] = 1
        selected_indices = np.where(selected == 1)[0]

        s = self.evaluate_defense(self.X[selected_indices])
        return self.X[selected_indices], best_fitness


class GA_UniversalPatch:
    def __init__(self,patches,s_model,train_loader,max_insert,numant,itermax):
        self.patches = patches
        self.s_model = s_model
        self.max_insert = max_insert
        self.numant = numant
        self.itermax = itermax
        self.train_loader = train_loader
        self.universal_solution = None
        self.best_solution = None
        self.best_fitness = None
    
    
    def run(self):
        # pdb.set_trace()
        # 加载模型
        model = torch.load(current_dir+'/../DLWF_pytorch/trained_model/'+self.s_model+'.pkl')
        model.eval()
        patch_list = []
        ant = Ant(model=model, numant=4, max_insert=self.max_insert,patches=self.patches,itermax=1)
        with tqdm(self.train_loader, unit="batch") as tepoch:
            for batch_x_tensor, batch_y_tensor in tqdm(self.train_loader, desc='Training'):
                batch_x_tensor = batch_x_tensor.float().to(device)
                batch_y_tensor = batch_y_tensor.float().to(device)
                u = ant.run(batch_x_tensor,batch_y_tensor)
                u = u.numpy().astype(int)
                patch_list.append(u)
        patch_list = np.array(patch_list).reshape(-1,2)
        patch_list = np.unique(patch_list, axis=0)
        patch_list = patch_list[patch_list[:, 0]<100]
        patch_list = patch_list[patch_list[:, 1]>self.max_insert/2]
        
        self.universal_solution = universal_solution(X=patch_list, patches=self.patches, model=model, train_loader=self.train_loader)
        self.best_solution, self.best_fitness = self.universal_solution.run_optimization()
        return self.best_solution, self.best_fitness