import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix

# ===== 复用你已有工程 =====
from train_relation_supcon_finetune import (
    TolerantMultiClassDataset,
    TolerantMultiViewEncoder,
    RelationProjector,
    VOCAB_SIZE_CFG,
    VOCAB_SIZE_AST,
    collate_fn
)

# ===== 3-class head 定义（必须与训练一致）=====
class ThreeClassHead(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(dim, dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(dim, 3)
        )

    def forward(self, x):
        return self.net(x)

# ===== 只保留 0/1/2 的 Dataset wrapper =====
class ThreeClassDataset(torch.utils.data.Dataset):
    def __init__(self, base_ds):
        self.items = [x for x in base_ds if x["label"] in (0, 1, 2)]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]

# =========================================================
# 3-class Test
# =========================================================
@torch.no_grad()
def test_3class(
    test_jsonl: str,
    finetune_ckpt: str = "best_relation_finetune.pt",
    head3_ckpt: str = "best_3class_head.pt",
    embedding_dim: int = 256,
    num_layers: int = 4
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # ---------- Dataset ----------
    base_test_ds = TolerantMultiClassDataset(test_jsonl)
    test_ds = ThreeClassDataset(base_test_ds)

    test_loader = DataLoader(
        test_ds,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_fn
    )

    print(f"3-class test samples: {len(test_ds)}")

    # ---------- Load encoder + relation ----------
    encoder = TolerantMultiViewEncoder(
        VOCAB_SIZE_CFG, VOCAB_SIZE_AST,
        embedding_dim, num_layers, device
    ).to(device)

    relation = RelationProjector(embedding_dim).to(device)

    ckpt = torch.load(finetune_ckpt, map_location=device)
    encoder.load_state_dict(ckpt["encoder"])
    relation.load_state_dict(ckpt["relation"])
    encoder.eval()
    relation.eval()

    # ---------- Load 3-class head ----------
    head3 = ThreeClassHead(embedding_dim).to(device)
    head3.load_state_dict(torch.load(head3_ckpt, map_location=device))
    head3.eval()

    print("✅ Loaded encoder + relation + 3-class head")

    # ---------- Testing ----------
    y_true, y_pred = [], []

    for batch in tqdm(test_loader, desc="Testing 3-class"):
        item = batch[0]
        label = item["label"]

        cfg_valid = item["cfg_valid"]
        ast_valid = item["ast_valid"]
        cfg_pack = item["cfg"]
        ast_pack = item["ast"]

        if cfg_pack:
            cfg_x1, cfg_e1, cfg_a1, cfg_x2, cfg_e2, cfg_a2 = cfg_pack
            cfg1 = (cfg_x1, cfg_e1, cfg_a1)
            cfg2 = (cfg_x2, cfg_e2, cfg_a2)
        else:
            cfg1 = cfg2 = None

        if ast_pack:
            ast_x1, ast_e1, ast_a1, ast_x2, ast_e2, ast_a2 = ast_pack
            ast1 = (ast_x1, ast_e1, ast_a1)
            ast2 = (ast_x2, ast_e2, ast_a2)
        else:
            ast1 = ast2 = None

        h1 = encoder.encode_program(cfg1, ast1, cfg_valid, ast_valid)
        h2 = encoder.encode_program(cfg2, ast2, cfg_valid, ast_valid)
        r = relation(h1, h2)

        pred = head3(r.unsqueeze(0)).argmax(dim=1).item()

        y_true.append(label)
        y_pred.append(pred)

    # ---------- Report ----------
    print("\n================ 3-Class Test Report ================\n")
    print(classification_report(
        y_true,
        y_pred,
        target_names=["SLSF", "SLDF", "DLSF"],
        digits=4
    ))
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))


# =========================================================
# Entry
# =========================================================
if __name__ == "__main__":
    test_3class(
        test_jsonl="generate_dataset/new_data/multiclass_dataset_test_lowread.jsonl",
        finetune_ckpt="best_relation_finetune.pt",
        head3_ckpt="best_3class_focalloss.pt"
    )
