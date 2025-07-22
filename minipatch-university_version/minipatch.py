import os
import time
import datetime
import argparse
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import tensorflow as tf
import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)),"../"))
from tensorflow.data import Dataset

from train_model import train_model
from perturb_utils import perturb_trace, patch_length
from perturb_utils import verify_perturb
from process_utils import load_trained_model, load_data
from process_utils import load_checkpoint, save_checkpoint, del_checkpoint
from dual_annealing import dual_annealing
from metrics import get_metrics

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# from adversarial_generation import *

# Basic settings
training = False        # [True False]
model = 'DF'            # ['AWF' 'DF' 'VarCNN']
dataset = 'AWF'     # ['Sirinam' 'Rimmer100' 'Rimmer200' 'Rimmer500' 'Rimmer900', 'AWF']
num_sites = -1
num_samples = -1
verify_model = None     # [None 'AWF' 'DF' 'VarCNN']
verify_data = None      # [None '3d' '10d' '2w' '4w' '6w']
verbose = 1             # [0 1 2]

# Hyperparameters for patch generation
patches = 500             # [1 2 4 8]


maxiter = 1000            # [30 40 ... 100]
maxquery = 1e5          # [1e1 1e2 ... 1e7]
threshold = 1           # [0.9 0.91 ... 1.0]

# Hyperparameters for Dual Annealing
initial_temp = 5230.
restart_temp_ratio = 2.e-5
visit = 2.62
accept = -1e3


import pdb
class Minipatch:
    """
    Minipatch implementation.
    """
    def __init__(self, model, traces, labels, names=None, verbose=0, evaluate=True):
        self.model = model
        try:
            self.input_size = model.input_shape[1]
            self.num_classes = model.output_shape[1]
        except AttributeError:
            self.input_size = traces.shape[-1]
            self.num_classes = len(np.unique(labels))
        # print(self.input_size, self.num_classes)
        print('traces:', traces.shape, 'labels:', labels.shape)

        self.traces = torch.tensor(traces)   #shape (124991, 1000, 1)
        self.classes = names
        self.labels = torch.tensor(labels)    #shape (124991, 100)
        self.verbose = verbose
        
        if evaluate:
            
            all_outputs = []
            count = 0
            for i in range(0,traces.shape[0],100):
                count = i
                # print(count)
                output = self.model(self.traces[i:i+100,:,:].to(device)).argmax(-1)#tensor会自动截断到合法范围
                all_outputs.append(output)
     
            self.preds = torch.cat(all_outputs).flatten().cpu().numpy()
          

            print('self.preds:', self.preds.shape)
            print('self.labels:', self.labels.shape)
            true_labels = self.labels.argmax(dim=1).cpu().numpy()    # shape: (124991,)
            preds = self.preds                       # shape: (124991,)

            #存储每个类别对应的下标
            self.correct = [
                np.where((preds == true_labels) & (true_labels == site_id))[0]
                for site_id in range(self.num_classes)
            ]

    def perturb_all(self, bounds, maxiter, maxquery, threshold):

        trace_ids = np.arange(len(self.traces))

        if self.verbose > 0:
            print('Perturbing all traces (%d)...' % len(trace_ids))

        result = self.perturb_website(trace_ids, bounds, maxiter, maxquery, threshold)
        return result




    def perturb_website(self, trace_ids,bounds, maxiter, maxquery, threshold):
        
        """
        self.traces shape:[124991, 1000, 1]
        Generate perturbation for traces of a website.
        """
        # pdb.set_trace()
        test_traces = self.traces[:,:,0]
        lengths = [(test_traces[i] != 0).sum().item() for i in trace_ids]
        length_bound = (1, np.percentile(lengths, 50))
        patches = bounds['patches']
        # print("patches:",patches)
        perturb_bounds = [length_bound] * patches

        start = time.perf_counter()

        # Format the objective and callback functions for Dual Annealing

        def objective_func_withbatch(perturbation, batch_size=64):

            total_confidence = 0.0
            traces = self.traces[trace_ids] # Access traces via closure
            num_batches = (len(traces) + batch_size - 1) // batch_size
            with torch.no_grad():
                for batch_idx in range(num_batches):
                    start = batch_idx * batch_size
                    end = min(start + batch_size, len(traces))
                    batch_traces = traces[start:end]  # Slice the traces
                    # Evaluate the objective function on the current batch
                    batch_confidence = self.predict_multi_classes(batch_traces, perturbation, self.labels[trace_ids][start:end])
                    total_confidence += batch_confidence * (end - start)  # Weight by batch size
                # Return the average confidence over all batches

            print("confidence:",total_confidence / len(traces)," total_confidence/traces:",total_confidence,"/",len(traces))
            return total_confidence / len(traces)

        def callback_func_withbatch(perturbation, f, context, batch_size=64):
            total_success = 0
            traces = self.traces[trace_ids].cpu().numpy()  # Access traces via closure
            num_batches = (len(traces) + batch_size - 1) // batch_size
            for batch_idx in range(num_batches):
                start = batch_idx * batch_size
                end = min(start + batch_size, len(traces))
                batch_traces = traces[start:end]  # Slice the traces
                # Evaluate perturb_success_batch on the current batch
                batch_success = self.perturb_success_batch(batch_traces, perturbation, self.labels[trace_ids][start:end], threshold, batch_size)
                total_success += batch_success * (end - start)  # Weight by batch size
            # Return the average success rate over all batches
            print("total_success:",total_success)#打印预测准确的样本的数
            return total_success / len(traces)

     
        perturb_result = dual_annealing(
            objective_func_withbatch, perturb_bounds,
            maxiter=maxiter,
            maxfun=maxquery,
            initial_temp=initial_temp,
            restart_temp_ratio=restart_temp_ratio,
            visit=visit,
            accept=accept,
            callback=callback_func_withbatch,
            no_local_search=False,
            disp=False)

        end = time.perf_counter()

        # Record optimization results
        perturbation = perturb_result.x.astype(int)
        # print(f'Result:', perturbation)
        iteration = perturb_result.nit
        execution = perturb_result.nfev
        duration = end - start

        # Apply the optimized perturbation
        perturbed_traces = perturb_trace(self.traces[trace_ids].cpu().numpy(), perturbation)
        # Note: model.predict() is much slower than model(training=False)

        batch_size = 512  # Adjust batch size based on GPU memory
        num_batches = (len(perturbed_traces) + batch_size - 1) // batch_size
        predictions = []
        with torch.no_grad():
            for batch_idx in range(num_batches):
                start = batch_idx * batch_size
                end = min(start + batch_size, len(perturbed_traces))
                batch_traces = perturbed_traces[start:end]  # Slice the traces
                # Get model predictions for the current batch
                batch_traces = torch.tensor(batch_traces).to(device)
                batch_predictions = self.model(batch_traces)
                predictions.append(batch_predictions.cpu().numpy())
        predictions = np.concatenate(predictions, axis=0)

        # Calculate some statistics to return from this function
        # pdb.set_trace()


        pred_class = predictions.argmax(axis=-1)


        true_labels = self.labels[trace_ids].argmax(-1)  # 获取真实标签列表
        success = [pred != true for pred, true in zip(pred_class, true_labels)] #防御成功的样本数量

        num_valid = len(trace_ids)
        num_success = sum(success)
        if num_success >= num_valid * threshold:
            successful = True
        else:
            successful = False
        # standard calculate
        TPR,FPR,F1,ACC,overall_ACC = get_metrics(self.labels[trace_ids].argmax(-1), pred_class)
        print('\033[31m Standard metrices:\033[0m (TPR,FPR,F1,ACC,overall_ACC)',
              np.array(TPR).mean(),np.array(FPR).mean(),np.array(F1).mean(),np.array(ACC).mean(),overall_ACC)

        # Result dictionary
        result = {
            'perturbation': perturbation.tolist(),
            'result':
              ''.join((str(np.array(TPR).mean()),str(np.array(FPR).mean()),str(np.array(F1).mean()),str(np.array(ACC).mean()),str(overall_ACC)))
            }

        if self.verbose > 0:
            print('%s - perturbe success rate: %.2f%% (%d/%d) - iter: %d (%d) - time: %.2fs' % (
                'Succeeded' if num_success >= num_valid * threshold else 'Failed', 100 * num_success / num_valid,
                num_success, num_valid, iteration, execution, duration))

        return result



    def predict_multi_classes(self, traces, perturbations, tar_classes):
        """
        The objective function of the optimization problem.
        Perturb traces and get the model confidence for each sample's target class.
        """

        perturbed_traces = perturb_trace(traces.numpy(), perturbations)

        perturbed_traces = torch.tensor(perturbed_traces)
        predictions = self.model(perturbed_traces.to(device))  # Shape: [B, Num_Classes]
        import torch.nn.functional as F

        predictions = F.softmax(predictions, dim=1)
        
        # mean_confidence = predictions[torch.arange(0,predictions.shape[0]),tar_classes.argmax(-1)].mean()
        pred = predictions.argmax(-1).cpu()
        mean_confidence = (pred==tar_classes.argmax(-1)).sum().item()/len(perturbed_traces)
        return mean_confidence


    def perturb_success_batch(self, traces, perturbation, target, threshold, batch_size=100):
        """
        Check if the perturbation is successful for the given traces.
        target: tensor shape:(512,100)
        """

        temp = target.numpy()
        perturbed_traces = perturb_trace(traces, perturbation)
        # Split perturbed_traces into batches
        num_batches = (len(perturbed_traces) + batch_size - 1) // batch_size
        all_predictions = []
        for batch_idx in range(num_batches):
            start = batch_idx * batch_size
            end = min(start + batch_size, len(perturbed_traces))
            batch_traces = torch.tensor(perturbed_traces[start:end]).to(device)  # Slice the traces
            
            batch_predictions = self.model(batch_traces)
            all_predictions.append(batch_predictions.cpu().detach().numpy())
        predictions = np.concatenate(all_predictions, axis=0)
        # Calculate success rate
        pred_class = predictions.argmax(axis=-1)
        success = (pred_class==temp.argmax(-1))  # Compare with true labels
        num_success = success.sum().item()
        return num_success / len(traces) 




if __name__ == '__main__':
    import os
    print('Current path:', os.getcwd().replace('\\','/'))

    parser = argparse.ArgumentParser(
        description='Minipatch: Undermining DNN-based Website Fingerprinting with Adversarial Patches')
    parser.add_argument('-t', '--train', action='store_true', default=training,
        help='Training DNN model for Deep Website Fingerprinting.')
    parser.add_argument('-m', '--model', default=model,
        help='Target DNN model. Supports ``AWF``, ``DF`` and ``VarCNN``.')
    parser.add_argument('-d', '--data', default=dataset,
        help='Website trace dataset. Supports ``Sirinam`` and ``Rimmer100/200/500/900``.')
    parser.add_argument('-nw', '--websites', type=int, default=num_sites,
        help='The number of websites to perturb. Take all websites if set to -1.')
    parser.add_argument('-ns', '--samples', type=int, default=num_samples,
        help='The number of trace samples to perturb. Take all samples if set to -1.')
    parser.add_argument('-vm', '--verify_model', default=verify_model,
        help='Validation Model. Default is the same as the target model.')
    parser.add_argument('-vd', '--verify_data', default=verify_data,
        help='Validation data. Default is the validation data. Supports ``3d/10d/2w/4w/6w`` with ``Rimmer200``.')
    parser.add_argument('--patches', type=int, default=patches,
        help='The number of perturbation patches.')
    parser.add_argument('--maxiter', type=int, default=maxiter,
        help='The maximum number of iteration.')
    parser.add_argument('--maxquery', type=int, default=maxquery,
        help='The maximum number of queries accessing the model.')
    parser.add_argument('--threshold', type=float, default=threshold,
        help='The threshold to determine perturbation success.')
    parser.add_argument('--verbose', type=int, default=verbose,
        help='Print out information. 0 = progress bar, 1 = one line per item, 2 = show perturb details.')

    # Parsing parameters
    args = parser.parse_args()
    training = args.train
    target_model = args.model
    dataset = args.data
    num_sites = args.websites
    num_samples = args.samples
    if args.verify_model is None:
        verify_model = target_model
    else:
        verify_model = args.verify_model
    if args.verify_data is None:
        verify_data = 'valid'
    else:
        verify_data = args.verify_data
    bounds = {
        'patches': args.patches
        }
    optim_maxiter = args.maxiter
    optim_maxquery = args.maxquery
    success_thres = args.threshold
    verbose = args.verbose


    result_dir = './results/%s_%s/' % (target_model.lower(), dataset.lower())
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    
    result_file = result_dir + '%dpatches_%dmaxiter_%dmaxquery_%dthreshold_%swebsites_%ssamples' % (
        bounds['patches'], optim_maxiter, optim_maxquery, success_thres * 100, 
        'all' if num_sites == -1 else str(num_sites), 'all' if num_samples == -1 else str(num_samples))


    print('==> Start perturbing websites...')

    from Save_model.ModelWrapper import ModelWrapper
    from utils.data import load_rimmer_dataset
    from model_1000 import VarCNN
    from model import *
    import json
    #加载模型
    # wf_model = VarCNN(1000,100).to(device)
    # wf_model.load_state_dict(torch.load('/root/autodl-tmp/HAAD-GA/DLWF_pytorch/trained_model/length_1000/varcnn_1000.pth'))
    wf_model = torch.load('/root/autodl-tmp/HAAD-GA/DLWF_pytorch/trained_model/Tor_DF.pkl')
    train_x, train_y, X_valid, y_valid, test_x, test_y = load_rimmer_dataset(input_size=5000, num_classes=100) #(124991, 1000, 1),(124991, 100)
    
    minipatch = Minipatch(wf_model, train_x, train_y, None, verbose)
    start_time = time.time()
    # pdb.set_trace()
    result = minipatch.perturb_all( bounds, optim_maxiter, optim_maxquery, success_thres)
    # print(result)
    with open('%s.json' % result_file,"w",encoding='utf-8') as f:
        
        f.write(str(result['perturbation']))
        f.write('\n')
        f.write(str(result['result']))
    
    print('\033[31m Time cost:\033[0m ', time.time() - start_time, 's')


