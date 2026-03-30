import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class WESADDataset(Dataset):
    def __init__(self, data_root, subject_ids, window_sec=60, overlap=0.5, target_fs=100, mode='multivariate'):
        """
        WESAD Multimodal Dataset Loader for Advanced DL.
        
        Args:
            data_root (str): Path to WESAD folder
            subject_ids (list): Subject IDs
            window_sec (int): Window size in seconds
            overlap (float): Overlap ratio (0.5 = 50%)
            target_fs (int): Downsample frequency (Hz)
            mode (str): 'multivariate' (Seq, Feat) or 'independent' (original behavior)
        """
        self.data_root = data_root
        self.subject_ids = subject_ids
        self.target_fs = target_fs
        self.window_size = int(window_sec * target_fs)
        self.step_size = int(self.window_size * (1 - overlap))
        self.mode = mode
        
        # Combined storage for multivariate (Samples, Seq, 4)
        self.data = []
        self.labels = []
        
        # Individual storage for support if needed (maintaining compatibility if possible)
        self.ecg, self.eda, self.emg, self.resp, self.temp = [], [], [], [], []
        
        self.label_map = {1: 0, 3: 1, 2: 2}
        self._load_subjects()

    def _load_subjects(self):
        from scipy.signal import resample
        for sid in self.subject_ids:
            file_path = os.path.join(self.data_root, sid, f"{sid}.pkl")
            if not os.path.exists(file_path): continue
                
            with open(file_path, 'rb') as f:
                data = pickle.load(f, encoding='latin1')
            
            # Original FS = 700
            orig_fs = 700
            
            labels_raw = data['label'].flatten()
            ecg_raw = data['signal']['chest']['ECG'].flatten()
            eda_raw = data['signal']['chest']['EDA'].flatten()
            emg_raw = data['signal']['chest']['EMG'].flatten()
            resp_raw = data['signal']['chest']['Resp'].flatten()
            temp_raw = data['signal']['chest']['Temp'].flatten()
            
            # Downsample Labels (Majority voting later or decimation)
            # Decimation for signals
            num_samples = int(len(labels_raw) * self.target_fs / orig_fs)
            
            ecg = resample(ecg_raw, num_samples)
            eda = resample(eda_raw, num_samples)
            emg = resample(emg_raw, num_samples)
            resp = resample(resp_raw, num_samples)
            temp = resample(temp_raw, num_samples)
            
            # Normalize Per Signal Per Subject
            def norm(x): return (x - np.mean(x)) / (np.std(x) + 1e-8)
            ecg, eda, emg, resp = norm(ecg), norm(eda), norm(emg), norm(resp)
            
            # Downsample labels by taking indices
            indices = np.linspace(0, len(labels_raw) - 1, num_samples).astype(int)
            labels_raw_ds = labels_raw[indices]
            resp = norm(resp)
            temp = norm(temp)
            
            # Segments per subject
            for i in range(0, num_samples - self.window_size, self.step_size):
                window_labels = labels_raw_ds[i : i + self.window_size]
                if len(window_labels) < self.window_size: continue
                
                # Assign label as majority of window
                counts = np.bincount(window_labels)
                final_label = np.argmax(counts)
                
                if final_label in self.label_map:
                    target_l = self.label_map[final_label]
                    
                    if self.mode == 'multivariate':
                        # Stack (Seq, 5)
                        feat = np.stack([ecg[i:i+self.window_size], 
                                         eda[i:i+self.window_size], 
                                         emg[i:i+self.window_size], 
                                         resp[i:i+self.window_size],
                                         temp[i:i+self.window_size]], axis=1)
                        self.data.append(feat)
                    else:
                        self.ecg.append(ecg[i:i+self.window_size])
                        self.eda.append(eda[i:i+self.window_size])
                        self.emg.append(emg[i:i+self.window_size])
                        self.resp.append(resp[i:i+self.window_size])
                        self.temp.append(temp[i:i+self.window_size])
                    
                    self.labels.append(target_l)
                    
        self.labels = torch.tensor(np.array(self.labels), dtype=torch.long)
        
        if self.mode == 'multivariate':
            self.data = torch.tensor(np.array(self.data), dtype=torch.float32)
        else:
            self.ecg = torch.tensor(np.array(self.ecg), dtype=torch.float32).unsqueeze(1)
            self.eda = torch.tensor(np.array(self.eda), dtype=torch.float32).unsqueeze(1)
            self.emg = torch.tensor(np.array(self.emg), dtype=torch.float32).unsqueeze(1)
            self.resp = torch.tensor(np.array(self.resp), dtype=torch.float32).unsqueeze(1)
            self.temp = torch.tensor(np.array(self.temp), dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if self.mode == 'multivariate':
            return self.data[idx], self.labels[idx]
        return self.ecg[idx], self.eda[idx], self.emg[idx], self.resp[idx], self.temp[idx], self.labels[idx]

if __name__ == "__main__":
    # Smoke Test — multivariate mode (default for advanced models)
    WESAD_PATH = r'c:\Users\nisha\OneDrive\Desktop\Quantum_Computing\MultiModal_Quantum_Fusion\WESAD'
    dataset = WESADDataset(WESAD_PATH, ['S2'], window_sec=60, target_fs=100, mode='multivariate')
    print("Dataset size:", len(dataset))
    if len(dataset) > 0:
        data, label = dataset[0]
        print("Shape (Seq, Features):", data.shape)
        print("Label:", label.item())
        
        loader = DataLoader(dataset, batch_size=4, shuffle=True)
        batch_data, batch_label = next(iter(loader))
        print("Batch data shape:", batch_data.shape)
        print("Batch label shape:", batch_label.shape)
