import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from wesad_dataset import WESADDataset
from advanced_models import LSTMModel, CNNLSTMModel, TransformerModel, ClassicalBaseline
import time
import json
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# All 15 WESAD subjects
ALL_SUBJECTS = [f'S{i}' for i in [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17]]

def get_loso_split(fold_idx):
    """
    LOSO split: Test = 1 subject, Val = 1 subject, Train = 13 subjects.
    Matches train_loso.py methodology exactly.
    """
    n = len(ALL_SUBJECTS)
    test_subject = ALL_SUBJECTS[fold_idx]
    val_subject  = ALL_SUBJECTS[(fold_idx + 1) % n]
    train_subjects = [s for s in ALL_SUBJECTS if s != test_subject and s != val_subject]
    return train_subjects, [val_subject], [test_subject]


def compute_class_weights(dataset, device):
    """Inverse-frequency class weights to handle class imbalance."""
    labels = dataset.labels.numpy()
    unique, counts = np.unique(labels, return_counts=True)
    n_total = len(labels)
    weights = np.ones(3)
    for cls, cnt in zip(unique, counts):
        weights[cls] = n_total / (3 * cnt)
    print(f"  Class weights: {[round(w, 3) for w in weights.tolist()]}")
    return torch.tensor(weights, dtype=torch.float32).to(device)


def evaluate(model, loader, device, criterion):
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0

    with torch.no_grad():
        for data, label in loader:
            data, label = data.to(device), label.to(device)
            outputs = model(data)
            total_loss += criterion(outputs, label).item()
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(label.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='macro', zero_division=0
    )
    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1, 2])

    return {
        "loss": total_loss / len(loader),
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": cm
    }


def train_one_fold(fold_idx, model_type, args, device):
    """Train and evaluate a single LOSO fold."""
    train_subs, val_subs, test_subs = get_loso_split(fold_idx)
    model_id = f"{model_type}_{args['window_sec']}s"
    OUTPUT_FOLDER = os.path.join("output", "advanced_loso", model_id, f"fold_{fold_idx}")
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    WESAD_PATH = r'c:\Users\nisha\OneDrive\Desktop\Quantum_Computing\MultiModal_Quantum_Fusion\WESAD'

    print(f"\n{'='*60}")
    print(f"FOLD {fold_idx}/14 | Model: {model_type.upper()} | Test: {test_subs[0]}")
    print(f"Train({len(train_subs)}): {train_subs}")
    print(f"Val(1): {val_subs}  |  Test(1): {test_subs}")
    print(f"{'='*60}")

    # Load data
    train_ds = WESADDataset(WESAD_PATH, train_subs, window_sec=args['window_sec'],
                            target_fs=args['target_fs'], mode='multivariate')
    val_ds   = WESADDataset(WESAD_PATH, val_subs,   window_sec=args['window_sec'],
                            target_fs=args['target_fs'], mode='multivariate')
    test_ds  = WESADDataset(WESAD_PATH, test_subs,  window_sec=args['window_sec'],
                            target_fs=args['target_fs'], mode='multivariate')

    print(f"  Samples — Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args['batch_size'], shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=args['batch_size'], shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=args['batch_size'], shuffle=False)

    # Class-weighted loss (defined after model selection below, with label_smoothing)
    class_weights = compute_class_weights(train_ds, device)

    # Model
    if model_type == 'lstm':
        model = LSTMModel(num_features=5, num_classes=3).to(device)
    elif model_type == 'cnnlstm':
        model = CNNLSTMModel(num_features=5, num_classes=3).to(device)
    elif model_type == 'transformer':
        model = TransformerModel(num_features=5, num_classes=3).to(device)
    elif model_type == 'baseline':
        model = ClassicalBaseline(num_features=5, num_classes=3).to(device)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    optimizer = optim.AdamW(model.parameters(), lr=args['lr'], weight_decay=args['weight_decay'])
    # Monitor val_F1 (mode=max) — must match the checkpoint criterion.
    # Patience=10 to avoid premature LR decay on noisy single-subject validation.
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=10
    )

    # Label smoothing (0.1) reduces overconfidence on training subjects,
    # which is the main cross-subject generalization killer.
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)

    best_val_f1 = -1.0  # Checkpoint on F1 — val_loss is noisy with 1-subject validation
    history = {"train_loss": [], "val_loss": [], "val_acc": [], "val_f1": [], "lr": []}

    for epoch in range(args['epochs']):
        start = time.time()
        model.train()
        total_loss = 0

        for data, label in train_loader:
            data, label = data.to(device), label.to(device)
            optimizer.zero_grad()
            loss = criterion(model(data), label)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()

        val_m = evaluate(model, val_loader, device, criterion)
        avg_train = total_loss / len(train_loader)
        lr_now = optimizer.param_groups[0]['lr']

        history["train_loss"].append(avg_train)
        history["val_loss"].append(val_m["loss"])
        history["val_acc"].append(val_m["accuracy"])
        history["val_f1"].append(val_m["f1"])
        history["lr"].append(lr_now)

        scheduler.step(val_m["f1"])  # Track F1, not loss

        print(f"  Ep {epoch+1:02d}/{args['epochs']} | "
              f"Loss: {avg_train:.4f} | "
              f"Val Loss: {val_m['loss']:.4f} | "
              f"Val F1: {val_m['f1']:.4f} | "
              f"LR: {lr_now:.1e} | "
              f"{time.time()-start:.1f}s")

        if val_m["f1"] > best_val_f1:
            best_val_f1 = val_m["f1"]
            torch.save(model.state_dict(), os.path.join(OUTPUT_FOLDER, "best_model.pth"))

    # -- Training curves per fold --
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ep_x = range(1, args['epochs'] + 1)
    axes[0].plot(ep_x, history["train_loss"], label="Train Loss", color='#E74C3C')
    axes[0].plot(ep_x, history["val_loss"],   label="Val Loss",   color='#3498DB')
    axes[0].set_title(f"Loss — Fold {fold_idx} (Test: {test_subs[0]})")
    axes[0].set_xlabel("Epoch"); axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].plot(ep_x, history["val_acc"], label="Val Accuracy", color='#2ECC71')
    axes[1].plot(ep_x, history["val_f1"],  label="Val F1-Macro", color='#9B59B6')
    axes[1].set_title(f"Val Metrics — Fold {fold_idx}")
    axes[1].set_xlabel("Epoch"); axes[1].legend(); axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_FOLDER, "training_curves.png"))
    plt.close()

    # -- Test evaluation --
    model.load_state_dict(torch.load(os.path.join(OUTPUT_FOLDER, "best_model.pth"), weights_only=False))
    test_m = evaluate(model, test_loader, device, criterion)

    print(f"\n  >>> Fold {fold_idx} Test ({test_subs[0]}): "
          f"Acc={test_m['accuracy']:.4f} | F1={test_m['f1']:.4f} | "
          f"P={test_m['precision']:.4f} | R={test_m['recall']:.4f}")

    # Confusion Matrix per fold
    class_names = ['No Stress', 'Low Stress', 'High Stress']
    plt.figure(figsize=(7, 5))
    sns.heatmap(test_m['confusion_matrix'], annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Confusion Matrix — Fold {fold_idx} ({test_subs[0]})')
    plt.ylabel('Ground Truth'); plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_FOLDER, "confusion_matrix.png"))
    plt.close()

    # Save fold results
    fold_result = {
        "fold": fold_idx,
        "test_subject": test_subs[0],
        "val_subject": val_subs[0],
        "train_subjects": train_subs,
        "test_metrics": {k: v.tolist() if hasattr(v, 'tolist') else v
                         for k, v in test_m.items()},
        "history": history
    }
    with open(os.path.join(OUTPUT_FOLDER, "results.json"), "w") as f:
        json.dump(fold_result, f, indent=4)

    return test_m


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",        type=str,   choices=['lstm', 'cnnlstm', 'transformer', 'baseline'], default='lstm')
    parser.add_argument("--epochs",       type=int,   default=50)
    parser.add_argument("--batch_size",   type=int,   default=64)
    parser.add_argument("--lr",           type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--window_sec",   type=int,   default=3)
    parser.add_argument("--target_fs",    type=int,   default=100)
    parser.add_argument("--start_fold",   type=int,   default=0,
                        help="Resume from a specific fold (0-14)")
    parser.add_argument("--demo",         action="store_true",
                        help="Run only fold 0 for quick verification")
    args = vars(parser.parse_args())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    print(f"Model: {args['model'].upper()} | Window: {args['window_sec']}s @ {args['target_fs']}Hz")

    n_folds = 1 if args['demo'] else 15
    all_test_metrics = []

    for fold in range(args['start_fold'], n_folds):
        m = train_one_fold(fold, args['model'], args, device)
        all_test_metrics.append({**m, "fold": fold,
                                  "test_subject": ALL_SUBJECTS[fold]})

    # -- Aggregate LOSO Summary --
    if len(all_test_metrics) > 1:
        avg_acc = np.mean([m['accuracy'] for m in all_test_metrics])
        avg_f1  = np.mean([m['f1']       for m in all_test_metrics])
        avg_pre = np.mean([m['precision'] for m in all_test_metrics])
        avg_rec = np.mean([m['recall']    for m in all_test_metrics])
        std_f1  = np.std([m['f1']        for m in all_test_metrics])

        model_id = f"{args['model']}_{args['window_sec']}s"
        summary_dir = os.path.join("output", "advanced_loso", model_id)
        summary = {
            "model": model_id,
            "folds_completed": len(all_test_metrics),
            "avg_accuracy":  avg_acc,
            "avg_f1_macro":  avg_f1,
            "std_f1_macro":  std_f1,
            "avg_precision": avg_pre,
            "avg_recall":    avg_rec,
            "per_fold": [{k: v.tolist() if hasattr(v, 'tolist') else v
                          for k, v in m.items()}
                         for m in all_test_metrics]
        }
        with open(os.path.join(summary_dir, "loso_summary.json"), "w") as f:
            json.dump(summary, f, indent=4)

        # Summary bar chart
        subjects    = [m['test_subject'] for m in all_test_metrics]
        test_accs   = [m['accuracy'] for m in all_test_metrics]
        test_f1s    = [m['f1']       for m in all_test_metrics]

        x = np.arange(len(subjects))
        plt.figure(figsize=(16, 6))
        plt.bar(x - 0.2, test_accs, 0.4, label='Test Accuracy', color='#4A90E2', alpha=0.85)
        plt.bar(x + 0.2, test_f1s,  0.4, label='Test F1-Macro', color='#50E3C2', alpha=0.85)
        plt.axhline(avg_acc, color='#4A90E2', linestyle='--', linewidth=1, alpha=0.6)
        plt.axhline(avg_f1,  color='#50E3C2', linestyle='--', linewidth=1, alpha=0.6)
        for i, (ac, f1) in enumerate(zip(test_accs, test_f1s)):
            plt.text(i - 0.2, ac + 0.01, f'{ac:.2f}', ha='center', va='bottom', fontsize=8)
            plt.text(i + 0.2, f1 + 0.01, f'{f1:.2f}', ha='center', va='bottom', fontsize=8)
        plt.xticks(x, subjects)
        plt.ylim(0, 1.1)
        plt.xlabel('Test Subject'); plt.ylabel('Score')
        plt.title(f'LOSO Results — {args["model"].upper()} '
                  f'(Avg Acc={avg_acc:.3f}, Avg F1={avg_f1:.3f}±{std_f1:.3f})')
        plt.legend(); plt.grid(axis='y', alpha=0.3); plt.tight_layout()
        plt.savefig(os.path.join(summary_dir, "loso_summary_chart.png"))
        plt.close()

        print(f"\n{'='*60}")
        print(f"FINAL LOSO RESULTS — {args['model'].upper()}")
        print(f"  Avg Accuracy:  {avg_acc:.4f}")
        print(f"  Avg F1-Macro:  {avg_f1:.4f} ± {std_f1:.4f}")
        print(f"  Avg Precision: {avg_pre:.4f}")
        print(f"  Avg Recall:    {avg_rec:.4f}")
        print(f"  Summary saved: output/advanced_loso/{args['model']}/loso_summary.json")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()
