# -*- coding: utf-8 -*-
"""
3-class head training + ablation (Full / w-o AST / w-o CFG)
- Saves BEST results to JSON for each run
"""

import os
import json
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix

from train_relation_supcon_finetune import (
    TolerantMultiClassDataset,
    TolerantMultiViewEncoder,
    RelationProjector,
    VOCAB_SIZE_CFG,
    VOCAB_SIZE_AST,
    collate_fn
)

# =========================================================
# 1) 3-class head
# =========================================================
class ThreeClassHead(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(dim, 3)
        )

    def forward(self, x):
        return self.net(x)

# =========================================================
# 2) Filter dataset to keep only labels in {0,1,2}
# =========================================================
class ThreeClassDataset(torch.utils.data.Dataset):
    def __init__(self, base_dataset):
        self.items = [x for x in base_dataset if x["label"] in (0, 1, 2)]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]

# =========================================================
# 3) Loss (optional focal loss)
# =========================================================
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None):
        super().__init__()
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss(weight=weight, reduction="none")

    def forward(self, logits, target):
        ce = self.ce(logits, target)
        pt = torch.exp(-ce)
        loss = ((1 - pt) ** self.gamma) * ce
        return loss.mean()

# =========================================================
# 4) Utilities: build cfg/ast packs with ablation switches
# =========================================================
def unpack_pair(item, use_cfg: bool, use_ast: bool):
    label = item["label"]

    cfg_valid = bool(item["cfg_valid"])
    ast_valid = bool(item["ast_valid"])

    cfg_pack = item["cfg"]
    ast_pack = item["ast"]

    # ---- CFG
    if use_cfg and cfg_pack:
        cfg_x1, cfg_e1, cfg_a1, cfg_x2, cfg_e2, cfg_a2 = cfg_pack
        cfg1 = (cfg_x1, cfg_e1, cfg_a1)
        cfg2 = (cfg_x2, cfg_e2, cfg_a2)
    else:
        cfg1 = cfg2 = None
        cfg_valid = False

    # ---- AST
    if use_ast and ast_pack:
        ast_x1, ast_e1, ast_a1, ast_x2, ast_e2, ast_a2 = ast_pack
        ast1 = (ast_x1, ast_e1, ast_a1)
        ast2 = (ast_x2, ast_e2, ast_a2)
    else:
        ast1 = ast2 = None
        ast_valid = False

    return label, cfg1, cfg2, ast1, ast2, cfg_valid, ast_valid

# =========================================================
# 5) Result saving helpers
# =========================================================
def _ensure_dir(path: str):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)

def save_best_result_json(
    json_path: str,
    setting: dict,
    best_epoch: int,
    best_macro_f1: float,
    report_dict: dict,
    confusion_mat,
    train_size: int,
    test_size: int,
    ckpt_path: str,
    head_path: str,
    relation_path: str
):
    """
    Save a single run's best result into JSON.
    """
    _ensure_dir(json_path)

    payload = {
        "setting": setting,
        "best_epoch": best_epoch,
        "best_macro_f1": float(best_macro_f1),
        "train_size_3class": int(train_size),
        "test_size_3class": int(test_size),
        "checkpoint_encoder_relation": ckpt_path,
        "saved_head_path": head_path,
        "saved_relation_path": relation_path,
        "classification_report": report_dict,          # sklearn output_dict
        "confusion_matrix": confusion_mat.tolist(),    # numpy -> list
        "timestamp": int(time.time())
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"✅ Saved BEST results to JSON: {json_path}")

# =========================================================
# 6) Train + Eval (supports ablation)
# =========================================================
def train_3class_head(
    train_jsonl: str,
    test_jsonl: str,
    finetune_ckpt: str = "best_relation_finetune.pt",
    embedding_dim: int = 256,
    num_layers: int = 4,
    batch_size: int = 32,
    epochs: int = 50,
    save_path: str = "best_3class_head.pt",
    use_cfg: bool = True,
    use_ast: bool = True,
    loss_type: str = "focal",   # "focal" or "ce"
    do_tsne: bool = False,      # 你要 t-SNE 继续开 True
    results_json_path: str = None
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n" + "=" * 80)
    print(f"Device: {device}")
    print(f"Ablation setting: use_cfg={use_cfg}, use_ast={use_ast}")
    print(f"Train JSONL: {train_jsonl}")
    print(f"Test  JSONL: {test_jsonl}")
    print("=" * 80)

    # ---------- Dataset ----------
    train_ds_full = TolerantMultiClassDataset(train_jsonl)
    test_ds_full = TolerantMultiClassDataset(test_jsonl)

    train_ds = ThreeClassDataset(train_ds_full)
    test_ds = ThreeClassDataset(test_ds_full)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )
    test_loader = DataLoader(
        test_ds, batch_size=1, shuffle=False, collate_fn=collate_fn
    )

    print(f"Train samples (3-class): {len(train_ds)}")
    print(f"Test  samples (3-class): {len(test_ds)}")

    # ---------- Load encoder + relation ----------
    encoder = TolerantMultiViewEncoder(
        VOCAB_SIZE_CFG, VOCAB_SIZE_AST,
        embedding_dim, num_layers, device
    ).to(device)

    relation = RelationProjector(embedding_dim).to(device)

    ckpt = torch.load(finetune_ckpt, map_location=device)
    encoder.load_state_dict(ckpt["encoder"])
    relation.load_state_dict(ckpt["relation"])
    print(f"✅ Loaded encoder+relation from {finetune_ckpt}")

    # ---------- Freeze encoder for fair ablation ----------
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False

    relation.train()
    for p in relation.parameters():
        p.requires_grad = True

    head = ThreeClassHead(embedding_dim).to(device)
    head.train()

    # ---------- Optimizer ----------
    head_lr = 1e-3
    relation_lr = 3e-4
    optim = torch.optim.AdamW(
        [
            {"params": head.parameters(), "lr": head_lr},
            {"params": relation.parameters(), "lr": relation_lr},
        ],
        weight_decay=0.01
    )

    # ---------- Loss ----------
    if loss_type.lower() == "focal":
        criterion = FocalLoss(gamma=2.0)
    else:
        criterion = nn.CrossEntropyLoss()

    best_f1 = -1.0
    best_epoch = -1
    best_report_text = None
    best_report_dict = None
    best_cm = None

    best_relation_path = save_path.replace(".pt", "_relation.pt")

    # =====================================================
    # Training loop
    # =====================================================
    for ep in range(1, epochs + 1):
        head.train()
        relation.train()

        total_loss = 0.0
        steps = 0

        pbar = tqdm(train_loader, desc=f"[3-Class] Epoch {ep}/{epochs}")
        for batch in pbar:
            pair_embs = []
            labels = []

            for item in batch:
                label, cfg1, cfg2, ast1, ast2, cfg_valid, ast_valid = unpack_pair(
                    item, use_cfg=use_cfg, use_ast=use_ast
                )

                with torch.no_grad():
                    h1 = encoder.encode_program(cfg1, ast1, cfg_valid, ast_valid)
                    h2 = encoder.encode_program(cfg2, ast2, cfg_valid, ast_valid)

                r = relation(h1, h2)
                pair_embs.append(r)
                labels.append(label)

            pair_embs = torch.stack(pair_embs).to(device)
            labels = torch.tensor(labels, dtype=torch.long, device=device)

            logits = head(pair_embs)
            loss = criterion(logits, labels)

            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(relation.parameters(), 1.0)
            optim.step()

            total_loss += loss.item()
            steps += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = total_loss / max(steps, 1)
        print(f"[Epoch {ep}] avg_loss={avg_loss:.6f}")

        # ---------------- Eval on TEST ----------------
        head.eval()
        relation.eval()

        y_true, y_pred = [], []
        with torch.no_grad():
            for batch in test_loader:
                item = batch[0]
                label, cfg1, cfg2, ast1, ast2, cfg_valid, ast_valid = unpack_pair(
                    item, use_cfg=use_cfg, use_ast=use_ast
                )

                h1 = encoder.encode_program(cfg1, ast1, cfg_valid, ast_valid)
                h2 = encoder.encode_program(cfg2, ast2, cfg_valid, ast_valid)
                r = relation(h1, h2)

                pred = head(r.unsqueeze(0)).argmax(dim=1).item()
                y_true.append(label)
                y_pred.append(pred)

        report_dict = classification_report(
            y_true, y_pred,
            target_names=["SLSF", "SLDF", "DLSF"],
            output_dict=True,
            digits=4,
            zero_division=0
        )
        f1_macro = report_dict["macro avg"]["f1-score"]
        print(f"[Epoch {ep}] 3-class macro-F1={f1_macro:.4f}")

        report_text = classification_report(
            y_true, y_pred,
            target_names=["SLSF", "SLDF", "DLSF"],
            digits=4,
            zero_division=0
        )
        cm = confusion_matrix(y_true, y_pred)

        # save best
        if f1_macro > best_f1:
            best_f1 = f1_macro
            best_epoch = ep
            best_report_text = report_text
            best_report_dict = report_dict
            best_cm = cm

            torch.save(head.state_dict(), save_path)
            torch.save(relation.state_dict(), best_relation_path)

            print(f"✨ Saved best head to {save_path}")
            print(f"✨ Saved best relation to {best_relation_path}")

    # =====================================================
    # Training finished
    # =====================================================
    print("\n===== Training finished =====")
    print(f"Best epoch: {best_epoch} | Best macro-F1: {best_f1:.4f}")
    print("\n================ BEST Model 3-Class Test Report ================\n")
    print(best_report_text)
    print("Confusion Matrix:")
    print(best_cm)

    # ---------- Save JSON ----------
    if results_json_path is None:
        # default: same name as save_path but .json
        results_json_path = save_path.replace(".pt", ".json")

    setting = {
        "use_cfg": use_cfg,
        "use_ast": use_ast,
        "loss_type": loss_type,
        "embedding_dim": embedding_dim,
        "num_layers": num_layers,
        "batch_size": batch_size,
        "epochs": epochs,
    }

    save_best_result_json(
        json_path=results_json_path,
        setting=setting,
        best_epoch=best_epoch,
        best_macro_f1=best_f1,
        report_dict=best_report_dict,
        confusion_mat=best_cm,
        train_size=len(train_ds),
        test_size=len(test_ds),
        ckpt_path=finetune_ckpt,
        head_path=save_path,
        relation_path=best_relation_path
    )

    # Optional t-SNE（如需继续用你原来的版本，可自行合并）
    if do_tsne:
        print("\n[Info] do_tsne=True but t-SNE code omitted here for brevity.")
        print("       You can keep your collect_relation_embeddings + tsne_visualize as before.")

    return {
        "best_epoch": best_epoch,
        "best_macro_f1": best_f1,
        "report_dict": best_report_dict,
        "confusion_matrix": best_cm
    }

# =========================================================
# 7) Run three experiments: Full / w-o AST / w-o CFG
# =========================================================
if __name__ == "__main__":
    TRAIN_JSONL = "generate_dataset/new_data/multiclass_dataset_train.jsonl"
    TEST_JSONL  = "generate_dataset/new_data/multiclass_dataset_test.jsonl"
    CKPT        = "best_relation_finetune.pt"

    # 1) Full
    train_3class_head(
        train_jsonl=TRAIN_JSONL,
        test_jsonl=TEST_JSONL,
        finetune_ckpt=CKPT,
        embedding_dim=256,
        num_layers=4,
        batch_size=32,
        epochs=50,
        save_path="best_3class_full.pt",
        use_cfg=True,
        use_ast=True,
        loss_type="focal",
        do_tsne=False,
        results_json_path="results/ablation_full.json"
    )
