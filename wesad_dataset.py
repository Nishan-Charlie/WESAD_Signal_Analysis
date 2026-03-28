import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class WESADDataset(Dataset):
    def __init__(self, data_root, subject_ids, window_size=700, step_size=350, transform=None):
        """
        WESAD Multimodal Dataset Loader.
        
        Args:
            data_root (str): Path to WESAD folder (containing S2, S3, etc.)
            subject_ids (list): List of subject IDs to include (e.g., ['S2', 'S3'])
            window_size (int): Temporal window length (700 = 1 sec at 700Hz)
            step_size (int): Step for sliding window (350 = 50% overlap)
            transform (bool): Whether to apply Z-score normalization per Subject
        """
        self.data_root = data_root
        self.subject_ids = subject_ids
        self.window_size = window_size
        self.step_size = step_size
        
        # Data storage
        self.ecg = []
        self.eda = []
        self.emg = []
        self.resp = []
        self.labels = []
        
        # Label Mapping
        # WESAD: 1: Baseline, 2: Stress, 3: Amusement
        # Target: 0: No Stress (Baseline), 1: Low Stress (Amusement), 2: High Stress (Stress)
        self.label_map = {1: 0, 3: 1, 2: 2}
        
        self._load_subjects()

    def _load_subjects(self):
        print(f"Loading Subjects: {self.subject_ids}...")
        for sid in self.subject_ids:
            file_path = os.path.join(self.data_root, sid, f"{sid}.pkl")
            if not os.path.exists(file_path):
                print(f"Warning: {file_path} not found. Skipping.")
                continue
                
            with open(file_path, 'rb') as f:
                data = pickle.load(f, encoding='latin1')
            
            # Extract chest signals (all @ 700Hz)
            # Shapes: (N, 1) or (N,)
            ecg_raw = data['signal']['chest']['ECG'].flatten()
            eda_raw = data['signal']['chest']['EDA'].flatten()
            emg_raw = data['signal']['chest']['EMG'].flatten()
            resp_raw = data['signal']['chest']['Resp'].flatten()
            labels_raw = data['label'].flatten()
            
            # Subject-level Normalization
            ecg_raw = (ecg_raw - np.mean(ecg_raw)) / (np.std(ecg_raw) + 1e-8)
            eda_raw = (eda_raw - np.mean(eda_raw)) / (np.std(eda_raw) + 1e-8)
            emg_raw = (emg_raw - np.mean(emg_raw)) / (np.std(emg_raw) + 1e-8)
            resp_raw = (resp_raw - np.mean(resp_raw)) / (np.std(resp_raw) + 1e-8)
            
            # Sliding Window Segmenter
            self._segment_data(ecg_raw, eda_raw, emg_raw, resp_raw, labels_raw)
            
        # Convert lists to tensors
        self.ecg = torch.tensor(np.array(self.ecg), dtype=torch.float32).unsqueeze(1)
        self.eda = torch.tensor(np.array(self.eda), dtype=torch.float32).unsqueeze(1)
        self.emg = torch.tensor(np.array(self.emg), dtype=torch.float32).unsqueeze(1)
        self.resp = torch.tensor(np.array(self.resp), dtype=torch.float32).unsqueeze(1)
        self.labels = torch.tensor(np.array(self.labels), dtype=torch.long)
        
        print(f"Finished loading. Total segments: {len(self.labels)}")

    def _segment_data(self, ecg, eda, emg, resp, label):
        n_samples = len(label)
        for i in range(0, n_samples - self.window_size, self.step_size):
            window_label = label[i : i + self.window_size]
            
            # Use the majority label in the window, or just the middle value
            # Standard WESAD practice: Only keep window if all labels are the same
            unique_labels = np.unique(window_label)
            if len(unique_labels) == 1 and unique_labels[0] in self.label_map:
                target_label = self.label_map[unique_labels[0]]
                
                self.ecg.append(ecg[i : i + self.window_size])
                self.eda.append(eda[i : i + self.window_size])
                self.emg.append(emg[i : i + self.window_size])
                self.resp.append(resp[i : i + self.window_size])
                self.labels.append(target_label)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.ecg[idx], self.eda[idx], self.emg[idx], self.resp[idx], self.labels[idx]

if __name__ == "__main__":
    # Smoke Test
    WESAD_PATH = r'c:\Users\nisha\OneDrive\Desktop\Quantum_Computing\MultiModal_Quantum_Fusion\WESAD'
    # Test with Subject S2
    dataset = WESADDataset(WESAD_PATH, ['S2'], window_size=700, step_size=700) # Use no overlap for faster test
    print("Dataset size:", len(dataset))
    if len(dataset) > 0:
        ecg, eda, emg, resp, label = dataset[0]
        print("Shapes - ECG:", ecg.shape, "EDA:", eda.shape, "EMG:", emg.shape, "Resp:", resp.shape)
        print("Label:", label.item())
        
        loader = DataLoader(dataset, batch_size=4, shuffle=True)
        batch = next(iter(loader))
        print("Batch label shape:", batch[4].shape)
