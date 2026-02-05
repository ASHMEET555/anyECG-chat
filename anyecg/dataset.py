import os
import wfdb
import ast
import random
import numpy as np
import pandas as pd
from scipy.signal import resample
import torch
from torch.utils.data import Dataset
from anyecg.utils import _labels_, _diag_labels_, _form_labels_, _rhythm_labels_, _sub_diag_labels_, _super_diag_labels_, _sub_diag_text_, _super_diag_text_
from anyecg.utils import _ludb_labels_, _csn_labels_, _cpsc_labels_
from anyecg.utils import _ptbxl_dir_, _csn_dir_, _cpsc_dir_

def get_ecg_from_path(path, sampling_freq):
    """
    Loads an ECG file using WFDB, resamples it, and reorders leads.
    Returns: Numpy array of shape (Length, Leads) -> e.g., (5000, 12)
    """
    try:
        if path.endswith('.hea') or path.endswith('.dat'):
            path=os.path.splitext(path)[0]
        ecg, meta_info = wfdb.rdsamp(path)
        

    
    # ecg = ecg[:5000] # (5000, 12)
    if sampling_freq != meta_info['fs']:
            ecg = resample(ecg, int(ecg.shape[0] * sampling_freq / meta_info['fs']), axis=0)
    # crop /pad to standart lenght 1000 samples 
    # note model handles dynamic length 
    leads_order_ref = [item.lower() for item in ['I', 'II', 'III', 'AVR', 'AVL', 'AVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']]
    leads_order = [item.lower() for item in meta_info['sig_name']]

    if set(leads_order_ref).issubset(set(leads_order)):
        indices= [leads_order.index(item) for item in leads_order_ref]
        ecg = ecg[:, indices]
        # 3. CRITICAL FIX: Normalize to [-1, 1] range
        # Without this, gradients explode and loss becomes NaN
    min_val = np.min(ecg)
    max_val = np.max(ecg)
        
    if max_val - min_val > 1e-6: # Avoid division by zero
        ecg = 2 * (ecg - min_val) / (max_val - min_val) - 1
    else:
        ecg = np.zeros_like(ecg) # Flat line signal

        # 4. Handle NaNs in the data itself
        # Sometimes raw files have missing values (NaNs). Replace them with 0.
        ecg = np.nan_to_num(ecg)

    return ecg
    # if leads_order != leads_order_ref:
    #     ecg = ecg[:, [leads_order.index(item) for item in leads_order_ref]]
    # return ecg
    except Exception as e:
        print(f"Error Loading ECG at {path}:{e}")
        return np.zeros((1000,12))
class ReportGenCollator:
    """
    Handles data collation for the Report Generation task (Stage 1 & 2).
    Converts raw JSON entries into (ECG Tensor, Message List) tuples.
    """
    def __init__(self, sampling_freq=100, base_ecg_dir=""):
        self.sampling_freq = sampling_freq
        self.base_ecg_dir = base_ecg_dir
        
    def __call__(self, batch):
        ecgs = []
        messages = []

        for item in batch:
            # 1. Load ECG 
            ecg_path = item.get('path', item.get('ecg_path'))
            if self.base_ecg_dir and ecg_path:
                ecg_path = os.path.join(self.base_ecg_dir, ecg_path)
            
            if ecg_path:
                ecg_data = get_ecg_from_path(ecg_path, self.sampling_freq)
            else:
                ecg_data = np.zeros((1000, 12)) # Fallback

            # Transpose to match model input (Leads, Length)
            ecg_data = ecg_data.T
            ecgs.append(ecg_data)

            # 2. Format text message chat template 
            report = item.get('report', '')

            # Standard instruction format
            msg = [
                {"role": "user", "content": "Please provide the report for the following ECG."},
                {"role": "assistant", "content": report}
            ]
            messages.append(msg)
            
        # Convert ECG list to tensor 
        try:
            ecgs_tensor = torch.tensor(np.array(ecgs), dtype=torch.float32)
        except:
            # Fallback if lengths differ
            ecgs_tensor = [torch.tensor(e, dtype=torch.float32) for e in ecgs]
            
        # CRITICAL FIX: Return the data!
        return ecgs_tensor, messages
            
class FinetuningDataset(Dataset):
    def __init__(self, dataset, dataset_subtype, ecg_transform, sampling_freq, split_fold, proportion = 1.0):
        self.dataset, self.dataset_subtype = dataset, dataset_subtype
        self.ecg_transform = ecg_transform
        self.sampling_freq = sampling_freq

        filename = f'/mnt/sda1/xxxx/datasets/ECG/clip_data/data/{dataset}.csv'
        df = pd.read_csv(filename)

        df = df[df['split_fold']==split_fold]
        # if split_fold == 'train' and proportion != 1.0:
        if proportion != 1.0:
            df = df.sample(frac=proportion)
        # get label columns name
        self._label_test_, self._text_test_, self._ecg_dir_ = self._get_label_text()

        # get label for sub and super diag
        if dataset == 'ptbxl' and dataset_subtype in ['sub-diag', 'super-diag']:
            label_column = 'sub_diag_labels' if dataset_subtype == 'sub-diag' else 'super_diag_labels'
            df[label_column] = df[label_column].apply(lambda x: ast.literal_eval(x))
            for label in self._label_test_:
                df[label] = df[label_column].apply(lambda x: 1 if label in x else 0)
        # for ptbxl subsets, only keep samples with at least one label
        if dataset == 'ptbxl': 
            df['label_len'] = df[self._label_test_].sum(axis=1)
            df = df[df['label_len']>0]
        self.df = df
            
    def _get_label_text(self):
        if self.dataset == 'ptbxl':
            _label_test_ = {'all': _labels_,
                            'diag': _diag_labels_,
                            'form': _form_labels_,
                            'rhythm': _rhythm_labels_,
                            'sub-diag': _sub_diag_labels_,
                            'super-diag': _super_diag_labels_}[self.dataset_subtype]
        elif self.dataset == 'ludb':
            _label_test_ = _ludb_labels_
        elif self.dataset == 'csn':
            _label_test_ = _csn_labels_
        elif self.dataset == 'cpsc':
            _label_test_ = _cpsc_labels_
        else:
            raise NotImplementedError
        
        if self.dataset == 'ptbxl' and self.dataset_subtype in ['sub-diag', 'super-diag']:
            _text_test_ = {
                'sub-diag': _sub_diag_text_,
                'super-diag': _super_diag_text_
            }[self.dataset_subtype]
        else:
            _text_test_ = _label_test_

        _ecg_dir_ = {
            'ptbxl': _ptbxl_dir_,
            'csn': _csn_dir_,
            'cpsc': _cpsc_dir_
        }[self.dataset]
        return _label_test_, _text_test_, _ecg_dir_
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        sample = self.df.iloc[index]
        ecg_path = os.path.join(self._ecg_dir_, sample['path'])
        ecg = get_ecg_from_path(ecg_path, self.sampling_freq)
        ecg = self.ecg_transform(ecg)
        label = sample[self._label_test_].values
        return ecg, label
    
class FinetuningCollator:
    def __call__(self, batch):
        ecgs = [item[0] for item in batch]
        ecgs = np.array(ecgs, dtype=np.float32)
        ecgs = torch.as_tensor(ecgs).permute(0, 2, 1)
        labels = np.array([item[1] for item in batch], dtype=np.float32)
        labels = torch.as_tensor(labels, dtype=torch.float32)
        return ecgs, labels
