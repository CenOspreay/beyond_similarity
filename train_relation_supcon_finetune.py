import os
import json
import random
from typing import List, Dict, Any, Optional
from collections import Counter, defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Sampler
from tqdm import tqdm

import networkx as nx
import pickle

from tree_sitter import Language, Parser
from tree_sitter_languages import get_language

from sklearn.metrics import classification_report, confusion_matrix

# =========================
# 你的工程依赖
# =========================
from build_uCFG.build_uir import build_uir
from build_uCFG.build_ucfg import build_ucfg_from_uir
from model import GMNnet  # 必须提供 encode(x_ids, edge_index, edge_attr)
from train_multiview_v2 import supervised_contrastive_loss


# =========================================================
# 0) 全局可复现
# =========================================================
def seed_everything(seed: int = 42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

seed_everything(42)

# =========================================================
# 1) CFG 特征编码
# =========================================================
CATEGORY_IDS = {
    "Subroutine": 0,
    "Assignment": 1,
    "Conditional": 2,
    "Loop": 3,
    "Return": 4,
    "Statement": 5,
}
NUM_CATS = len(CATEGORY_IDS)
LABEL_LEN = 11

def encode_label_bits(label_list: List[int]) -> int:
    code = 0
    for i, b in enumerate(label_list):
        if b:
            code |= (1 << i)
    return code

def node_feature_id(node: Dict[str, Any]) -> int:
    cat = node.get("category", "Statement")
    cat_id = CATEGORY_IDS.get(cat, CATEGORY_IDS["Statement"])
    pattern_id = encode_label_bits(node.get("label", [0] * LABEL_LEN))
    return cat_id * (1 << LABEL_LEN) + pattern_id

VOCAB_SIZE_CFG = NUM_CATS * (1 << LABEL_LEN)  # 12288
MAX_CFG_NODES = 1200

def ucfg_to_graph_data(ucfg: Dict[str, Any]):
    nodes = ucfg.get("nodes", [])
    edges = ucfg.get("edges", [])

    if len(nodes) == 0:
        return [], [[], []], None

    if len(nodes) > MAX_CFG_NODES:
        nodes = nodes[:MAX_CFG_NODES]
        valid_ids = {n["id"] for n in nodes if "id" in n}
        edges = [(u, v, t) for (u, v, t) in edges if (u in valid_ids and v in valid_ids)]

    id2idx = {}
    for idx, n in enumerate(nodes):
        if "id" in n:
            id2idx[n["id"]] = idx

    x_ids = [node_feature_id(n) for n in nodes]

    src, dst = [], []
    for u, v, t in edges:
        if u in id2idx and v in id2idx:
            src.append(id2idx[u])
            dst.append(id2idx[v])

    edge_index = [src, dst] if len(src) > 0 else [[], []]
    return x_ids, edge_index, None

# =========================================================
# 2) AST 特征编码（递增 node_id，稳定唯一）
# =========================================================
AST_VOCAB_DICT_PATH = "generate_dataset/new_data/vocblen-mix-400-graph_mode-more_data-10-1-pruning-less-0.4-full-dict.pkl"
with open(AST_VOCAB_DICT_PATH, "rb") as f:
    vocab_dict: Dict[str, int] = pickle.load(f)

UNK_ID = vocab_dict.get("<unk>", 0)
VOCAB_SIZE_AST = max(vocab_dict.values()) + 1
MAX_AST_NODES = 3000

# tree-sitter C# 支持（可选）
SO_PATH = "build_uCFG/build/my-languages.so"
CS_REPO = "build_uCFG/build/tree-sitter-c-sharp"
if not os.path.exists(SO_PATH):
    Language.build_library(SO_PATH, [CS_REPO])
CS_LANGUAGE = Language(SO_PATH, "c_sharp")

language_dict = {"java": 4, "py": 2, "c": 1, "cpp": 3, "cs": 5}
def get_lang_id(cat: str) -> int:
    return language_dict.setdefault(cat, len(language_dict) + 1)

def parse_code_to_ast_graph(code: str, lang_cat: str) -> Optional[nx.DiGraph]:
    lang_id = get_lang_id(lang_cat)
    parser = Parser()

    if lang_id == 1:
        parser.set_language(get_language("c"))
    elif lang_id == 3:
        parser.set_language(get_language("cpp"))
    elif lang_id == 4:
        parser.set_language(get_language("java"))
    elif lang_id == 2:
        parser.set_language(get_language("python"))
    else:
        parser.set_language(CS_LANGUAGE)

    try:
        tree = parser.parse(code.encode("utf8"))
        root = tree.root_node

        graph = nx.DiGraph()
        node_counter = 0

        def traverse(node, parent_id=None):
            nonlocal node_counter
            node_id = node_counter
            node_counter += 1

            graph.add_node(node_id, label=node.type)
            if parent_id is not None:
                graph.add_edge(parent_id, node_id)

            if node_counter >= MAX_AST_NODES:
                return

            for ch in node.children:
                if node_counter >= MAX_AST_NODES:
                    break
                traverse(ch, node_id)

        traverse(root)
        return graph
    except Exception:
        return None

def ast_graph_to_data(graph: nx.DiGraph):
    if graph is None or graph.number_of_nodes() == 0:
        return [], [[], []], []

    if graph.number_of_nodes() > MAX_AST_NODES:
        nodes = list(graph.nodes())[:MAX_AST_NODES]
        graph = graph.subgraph(nodes).copy()

    nodes = list(graph.nodes())
    node2idx = {n: i for i, n in enumerate(nodes)}

    x_ids = [vocab_dict.get(graph.nodes[n].get("label", "<unk>"), UNK_ID) for n in nodes]

    src, dst = [], []
    for u, v in graph.edges():
        if u in node2idx and v in node2idx:
            src.append(node2idx[u])
            dst.append(node2idx[v])

    edge_index = [src, dst] if len(src) > 0 else [[], []]
    edge_attr = [0] * len(src)
    return x_ids, edge_index, edge_attr

# =========================================================
# 3) 容错 Dataset：至少一个视图有效就保留
# =========================================================
class TolerantMultiClassDataset(Dataset):
    """
    JSONL 格式：
      Code1, Code2, Category1, Category2, Label(1~4)
    """
    def __init__(self, jsonl_path: str, max_samples: Optional[int] = None):
        self.items: List[dict] = []
        self.stats = Counter()
        self.label2indices = defaultdict(list)

        with open(jsonl_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            if max_samples is not None:
                lines = lines[:max_samples]

            for idx, line in enumerate(tqdm(lines, desc=f"Loading {os.path.basename(jsonl_path)}")):
                data = json.loads(line.strip())

                code1 = data["Code1"]
                code2 = data["Code2"]
                lang1 = data["Category1"]
                lang2 = data["Category2"]
                label = int(data["Label"]) - 1  # 0~3

                # ---- CFG ----
                cfg_ok = True
                cfg_pack = None
                try:
                    uir1 = build_uir(code1, lang1)
                    uir2 = build_uir(code2, lang2)
                    if len(uir1) == 0 or len(uir2) == 0:
                        cfg_ok = False
                        self.stats["cfg_uir_empty"] += 1
                    else:
                        ucfg1 = build_ucfg_from_uir(uir1)
                        ucfg2 = build_ucfg_from_uir(uir2)
                        cfg_x1, cfg_e1, cfg_a1 = ucfg_to_graph_data(ucfg1)
                        cfg_x2, cfg_e2, cfg_a2 = ucfg_to_graph_data(ucfg2)
                        if len(cfg_x1) == 0 or len(cfg_x2) == 0:
                            cfg_ok = False
                            self.stats["cfg_empty"] += 1
                        else:
                            cfg_pack = (cfg_x1, cfg_e1, cfg_a1, cfg_x2, cfg_e2, cfg_a2)
                except Exception:
                    cfg_ok = False
                    self.stats["cfg_exception"] += 1

                # ---- AST ----
                ast_ok = True
                ast_pack = None
                try:
                    g1 = parse_code_to_ast_graph(code1, lang1)
                    g2 = parse_code_to_ast_graph(code2, lang2)
                    if g1 is None or g2 is None:
                        ast_ok = False
                        self.stats["ast_parse_fail"] += 1
                    else:
                        ast_x1, ast_e1, ast_a1 = ast_graph_to_data(g1)
                        ast_x2, ast_e2, ast_a2 = ast_graph_to_data(g2)
                        if len(ast_x1) == 0 or len(ast_x2) == 0:
                            ast_ok = False
                            self.stats["ast_empty"] += 1
                        else:
                            ast_pack = (ast_x1, ast_e1, ast_a1, ast_x2, ast_e2, ast_a2)
                except Exception:
                    ast_ok = False
                    self.stats["ast_exception"] += 1

                if (not cfg_ok) and (not ast_ok):
                    self.stats["drop_both_invalid"] += 1
                    continue

                item = {
                    "cfg": cfg_pack,
                    "ast": ast_pack,
                    "cfg_valid": 1 if cfg_ok else 0,
                    "ast_valid": 1 if ast_ok else 0,
                    "lang1": lang1,
                    "lang2": lang2,
                    "code1": code1,
                    "code2": code2,
                    "label": label,
                }
                self.items.append(item)
                self.label2indices[label].append(len(self.items) - 1)
                self.stats["kept"] += 1

        print("\n========== Dataset Stats ==========")
        print(f"File: {jsonl_path}")
        print(f"Raw lines: {len(lines)}")
        print(f"Kept: {self.stats['kept']}")
        print(f"Drop (both invalid): {self.stats['drop_both_invalid']}")
        for k, v in self.stats.items():
            if k not in ["kept"]:
                print(f"{k}: {v}")
        print("Label distribution:", {k: len(v) for k, v in self.label2indices.items()})
        print("===================================\n")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]

def collate_fn(batch):
    return batch

# =========================================================
# 4) BalancedBatchSampler：保证每类至少出现 >=2
# =========================================================
class BalancedBatchSampler(Sampler[List[int]]):
    """
    每个 batch 采样 n_classes 个类别，每类采样 n_samples 个样本
    batch_size = n_classes * n_samples
    """
    def __init__(self, dataset: TolerantMultiClassDataset, n_classes: int, n_samples: int, drop_last: bool = True):
        self.dataset = dataset
        self.n_classes = n_classes
        self.n_samples = n_samples
        self.drop_last = drop_last

        self.labels = list(dataset.label2indices.keys())
        # 过滤掉样本数不足的类（避免死循环）
        self.labels = [c for c in self.labels if len(dataset.label2indices[c]) >= n_samples]
        if len(self.labels) < n_classes:
            raise ValueError(f"Not enough classes with >= {n_samples} samples. Have {len(self.labels)} classes.")

        self.num_batches = len(dataset) // (n_classes * n_samples)

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        for _ in range(self.num_batches):
            chosen_classes = random.sample(self.labels, self.n_classes)
            batch = []
            for c in chosen_classes:
                batch.extend(random.sample(self.dataset.label2indices[c], self.n_samples))
            random.shuffle(batch)
            yield batch

# =========================================================
# 5) SupCon loss（监督式对比学习）
# =========================================================
def dldf_aware_supcon_loss(
    features: torch.Tensor,   # [B, D]
    labels: torch.Tensor,     # [B]
    temperature: float = 0.1,
    dldf_label: int = 3
):
    """
    SupCon for open-set:
    - only labels {0,1,2} have positives
    - DLDF (label=3) only appears as negative
    """
    device = features.device
    feats = F.normalize(features, dim=1)
    labels = labels.view(-1)

    B = feats.size(0)
    sim = torch.matmul(feats, feats.T) / temperature

    logits_mask = torch.ones((B, B), device=device) - torch.eye(B, device=device)
    sim = sim * logits_mask

    is_known = labels != dldf_label

    # 正样本：同类 + 都是 known
    pos_mask = (
        (labels.unsqueeze(0) == labels.unsqueeze(1)) &
        is_known.unsqueeze(0) &
        is_known.unsqueeze(1)
    ).float() * logits_mask

    exp_sim = torch.exp(sim) * logits_mask
    log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)

    pos_sum = (pos_mask * log_prob).sum(dim=1)
    pos_cnt = pos_mask.sum(dim=1)

    # 只在 known 且有正样本的 anchor 上算 loss
    valid = (pos_cnt > 0) & is_known
    if valid.sum() == 0:
        return torch.zeros((), device=device, requires_grad=True)

    loss = -(pos_sum[valid] / pos_cnt[valid]).mean()
    return loss


class KnownDLDFBatchSampler(Sampler[List[int]]):
    """
    每个 batch:
      - SLSF / SLDF / DLSF 各 k 个
      - DLDF m 个（hard negatives）
    """
    def __init__(
        self,
        dataset: TolerantMultiClassDataset,
        k: int = 3,
        m: int = 4
    ):
        self.dataset = dataset
        self.k = k
        self.m = m

        self.idx_by_label = dataset.label2indices
        for c in [0, 1, 2, 3]:
            if c not in self.idx_by_label:
                raise ValueError(f"Label {c} missing in dataset")

        self.num_batches = min(
            len(self.idx_by_label[0]) // k,
            len(self.idx_by_label[1]) // k,
            len(self.idx_by_label[2]) // k,
            len(self.idx_by_label[3]) // m,
        )

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        for _ in range(self.num_batches):
            batch = []
            for c in [0, 1, 2]:
                batch.extend(random.sample(self.idx_by_label[c], self.k))
            batch.extend(random.sample(self.idx_by_label[3], self.m))
            random.shuffle(batch)
            yield batch


# =========================================================
# 6) Encoder + Relation Network：pair embedding = g([h1,h2])
# =========================================================
class TolerantMultiViewEncoder(nn.Module):
    """
    给定单个程序的 CFG/AST（可能缺失），输出 program embedding [D]
    """
    def __init__(self, vocab_cfg: int, vocab_ast: int, dim: int, layers: int, device: torch.device):
        super().__init__()
        self.device = device
        self.dim = dim

        self.cfg_encoder = GMNnet(vocab_cfg, dim, layers, device).to(device)
        self.ast_encoder = GMNnet(vocab_ast, dim, layers, device).to(device)

        # mask-aware gate：输入 [h_cfg, h_ast, cfg_mask, ast_mask] -> alpha
        self.gate = nn.Sequential(
            nn.Linear(dim * 2 + 2, dim),
            nn.ReLU(),
            nn.Linear(dim, 1)
        )

    def _zero(self):
        return torch.zeros(self.dim, device=self.device)

    def encode_program(self, cfg_data, ast_data, cfg_valid: int, ast_valid: int) -> torch.Tensor:
        # CFG
        if cfg_valid and cfg_data is not None:
            cfg_x, cfg_e, cfg_a = cfg_data
            h_cfg = self.cfg_encoder.encode(cfg_x, cfg_e, cfg_a)
            if h_cfg.dim() > 1:
                h_cfg = h_cfg.view(-1)
        else:
            h_cfg = self._zero()

        # AST
        if ast_valid and ast_data is not None:
            ast_x, ast_e, ast_a = ast_data
            h_ast = self.ast_encoder.encode(ast_x, ast_e, ast_a)
            if h_ast.dim() > 1:
                h_ast = h_ast.view(-1)
        else:
            h_ast = self._zero()

        cfg_m = torch.tensor([float(cfg_valid)], device=self.device)
        ast_m = torch.tensor([float(ast_valid)], device=self.device)
        gate_in = torch.cat([h_cfg, h_ast, cfg_m, ast_m], dim=0)
        alpha = torch.sigmoid(self.gate(gate_in))  # [1]

        h = alpha * h_cfg + (1 - alpha) * h_ast
        return h

class RelationProjector(nn.Module):
    """
    Relation Network: r = g([h1,h2]) -> [D]
    """
    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim)
        )

    def forward(self, h1: torch.Tensor, h2: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([h1, h2], dim=-1))

class PairModel(nn.Module):
    """
    统一模型：
      - encoder: program embedding
      - relation: pair embedding
      - classifier: pair -> 4-class logits
    """
    def __init__(self, encoder: TolerantMultiViewEncoder, relation: RelationProjector, dim: int, num_classes: int = 4):
        super().__init__()
        self.encoder = encoder
        self.relation = relation
        self.classifier = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(dim, num_classes)
        )

    def encode_pair_batch(self, batch) -> (torch.Tensor, torch.Tensor):
        """
        batch: list[dict]
        return: pair_emb [B,D], labels [B]
        """
        pair_embs = []
        labels = []
        for item in batch:
            label = item["label"]
            cfg_valid = item["cfg_valid"]
            ast_valid = item["ast_valid"]

            cfg_pack = item["cfg"]
            ast_pack = item["ast"]

            if cfg_pack is not None:
                cfg_x1, cfg_e1, cfg_a1, cfg_x2, cfg_e2, cfg_a2 = cfg_pack
                cfg1 = (cfg_x1, cfg_e1, cfg_a1)
                cfg2 = (cfg_x2, cfg_e2, cfg_a2)
            else:
                cfg1 = None
                cfg2 = None

            if ast_pack is not None:
                ast_x1, ast_e1, ast_a1, ast_x2, ast_e2, ast_a2 = ast_pack
                ast1 = (ast_x1, ast_e1, ast_a1)
                ast2 = (ast_x2, ast_e2, ast_a2)
            else:
                ast1 = None
                ast2 = None

            h1 = self.encoder.encode_program(cfg1, ast1, cfg_valid, ast_valid)
            h2 = self.encoder.encode_program(cfg2, ast2, cfg_valid, ast_valid)

            r = self.relation(h1, h2)  # [D]
            pair_embs.append(r)
            labels.append(label)

        pair_embs = torch.stack(pair_embs, dim=0)
        labels = torch.tensor(labels, dtype=torch.long, device=pair_embs.device)
        return pair_embs, labels

    def forward(self, batch):
        pair_embs, labels = self.encode_pair_batch(batch)
        logits = self.classifier(pair_embs)
        return logits, labels, pair_embs

def known_only_supcon_loss(
    features: torch.Tensor,   # [B, D]
    labels: torch.Tensor,     # [B]
    temperature: float = 0.1,
):
    """
    只对 label ∈ {0,1,2} 计算 SupCon
    不包含 DLDF（label=3）
    """
    device = features.device

    mask = labels < 3
    if mask.sum() < 2:
        return torch.zeros((), device=device, requires_grad=True)

    feats = F.normalize(features[mask], dim=1)
    labs  = labels[mask]

    B = feats.size(0)
    sim = torch.matmul(feats, feats.T) / temperature

    logits_mask = torch.ones((B, B), device=device) - torch.eye(B, device=device)
    sim = sim * logits_mask

    pos_mask = (labs.unsqueeze(0) == labs.unsqueeze(1)).float() * logits_mask

    exp_sim = torch.exp(sim) * logits_mask
    log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)

    pos_sum = (pos_mask * log_prob).sum(dim=1)
    pos_cnt = pos_mask.sum(dim=1)

    valid = pos_cnt > 0
    if valid.sum() == 0:
        return torch.zeros((), device=device, requires_grad=True)

    loss = -(pos_sum[valid] / pos_cnt[valid]).mean()
    return loss


# =========================================================
# 7) SupCon 预训练
# =========================================================
def train_supcon(
    train_jsonl: str,
    embedding_dim: int = 256,
    num_layers: int = 4,
    n_classes_per_batch: int = 4,
    n_samples_per_class: int = 4,
    lr: float = 1e-3,
    epochs: int = 30,
    temperature: float = 0.1,
    patience: int = 8,
    save_path: str = "best_relation_supcon.pt",
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    dataset = TolerantMultiClassDataset(train_jsonl)
    sampler = KnownDLDFBatchSampler(
        dataset,
        k=4,  # 每个 known 类 4 个
        m=2  # DLDF 2 个
    )
    loader = DataLoader(dataset, batch_sampler=sampler, collate_fn=collate_fn)

    encoder = TolerantMultiViewEncoder(VOCAB_SIZE_CFG, VOCAB_SIZE_AST, embedding_dim, num_layers, device).to(device)
    relation = RelationProjector(embedding_dim, dropout=0.1).to(device)

    # SupCon 阶段：只需要 pair embedding（encoder+relation）
    params = list(encoder.parameters()) + list(relation.parameters())
    optim = torch.optim.Adam(params, lr=lr)

    best = float("inf")
    bad = 0

    for ep in range(1, epochs + 1):
        encoder.train()
        relation.train()

        total = 0.0
        steps = 0

        pbar = tqdm(loader, desc=f"[SupCon] Epoch {ep}/{epochs}", unit="batch")
        for batch in pbar:
            # 计算 pair embedding
            pair_embs = []
            labels = []
            for item in batch:
                label = item["label"]
                cfg_valid = item["cfg_valid"]
                ast_valid = item["ast_valid"]

                cfg_pack = item["cfg"]
                ast_pack = item["ast"]

                if cfg_pack is not None:
                    cfg_x1, cfg_e1, cfg_a1, cfg_x2, cfg_e2, cfg_a2 = cfg_pack
                    cfg1 = (cfg_x1, cfg_e1, cfg_a1)
                    cfg2 = (cfg_x2, cfg_e2, cfg_a2)
                else:
                    cfg1 = None
                    cfg2 = None

                if ast_pack is not None:
                    ast_x1, ast_e1, ast_a1, ast_x2, ast_e2, ast_a2 = ast_pack
                    ast1 = (ast_x1, ast_e1, ast_a1)
                    ast2 = (ast_x2, ast_e2, ast_a2)
                else:
                    ast1 = None
                    ast2 = None

                h1 = encoder.encode_program(cfg1, ast1, cfg_valid, ast_valid)
                h2 = encoder.encode_program(cfg2, ast2, cfg_valid, ast_valid)
                r = relation(h1, h2)
                pair_embs.append(r)
                labels.append(label)

            pair_embs = torch.stack(pair_embs, dim=0).to(device)
            labels = torch.tensor(labels, dtype=torch.long, device=device)

            loss_known = known_only_supcon_loss(
                pair_embs,
                labels,
                temperature=temperature
            )

            loss_repel = dldf_aware_supcon_loss(
                pair_embs,
                labels,
                temperature=temperature,
                dldf_label=3
            )

            lambda_repel = 0.3  # ⭐ 关键超参，不要 >0.5
            loss = loss_known + lambda_repel * loss_repel

            optim.zero_grad()
            loss.backward()
            optim.step()

            total += loss.item()
            steps += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg = total / max(steps, 1)
        print(f"[SupCon] Epoch {ep} avg_loss={avg:.6f}")

        if avg < best:
            best = avg
            bad = 0
            torch.save({
                "encoder": encoder.state_dict(),
                "relation": relation.state_dict(),
                "embedding_dim": embedding_dim,
                "num_layers": num_layers
            }, save_path)
            print(f"✨ Saved best SupCon to {save_path} (loss={best:.6f})")
        else:
            bad += 1
            print(f"patience {bad}/{patience}")
            if bad >= patience:
                print("⛔ SupCon early stopping.")
                break

    print("SupCon done. Best loss:", best)

# =========================================================
# 8) Finetune：端到端训练（encoder+relation+classifier）
#    可选：加 SupCon 辅助项（建议先不开：lambda_supcon=0.0）
# =========================================================
@torch.no_grad()
def evaluate_pair_model(model: PairModel, loader: DataLoader, device: torch.device, title: str = "Eval"):
    model.eval()
    y_true, y_pred = [], []

    for batch in tqdm(loader, desc=f"[{title}]", unit="sample"):
        logits, labels, _ = model(batch)
        pred = logits.argmax(dim=1).cpu().tolist()
        y_pred.extend(pred)
        y_true.extend(labels.cpu().tolist())

    target_names = ["SLSF", "SLDF", "DLSF", "DLDF"]
    print("\n================ Classification Report ================\n")
    print(classification_report(y_true, y_pred, target_names=target_names, digits=4))
    print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))

def finetune(
    train_jsonl: str,
    test_jsonl: str,
    supcon_ckpt: str = "best_relation_supcon.pt",
    embedding_dim: int = 256,
    num_layers: int = 4,
    batch_size: int = 16,
    lr: float = 1e-4,
    epochs: int = 20,
    lambda_supcon: float = 0.0,  # 如需“边分类边保持表示”，可设 0.05~0.2
    temperature: float = 0.1,
    save_path: str = "best_relation_finetune.pt"
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    train_ds = TolerantMultiClassDataset(train_jsonl)
    test_ds  = TolerantMultiClassDataset(test_jsonl)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    test_loader  = DataLoader(test_ds,  batch_size=1,         shuffle=False, collate_fn=collate_fn)

    # build model
    encoder = TolerantMultiViewEncoder(VOCAB_SIZE_CFG, VOCAB_SIZE_AST, embedding_dim, num_layers, device).to(device)
    relation = RelationProjector(embedding_dim, dropout=0.1).to(device)
    model = PairModel(encoder, relation, embedding_dim, num_classes=4).to(device)

    # load supcon weights
    ckpt = torch.load(supcon_ckpt, map_location=device)
    model.encoder.load_state_dict(ckpt["encoder"])
    model.relation.load_state_dict(ckpt["relation"])
    print(f"✅ Loaded SupCon ckpt: {supcon_ckpt}")

    optim = torch.optim.Adam(model.parameters(), lr=lr)
    ce = nn.CrossEntropyLoss()

    best_f1_macro = -1.0

    for ep in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        steps = 0

        pbar = tqdm(train_loader, desc=f"[Finetune] Epoch {ep}/{epochs}", unit="batch")
        for batch in pbar:
            logits, labels, pair_embs = model(batch)

            loss_cls = ce(logits, labels)

            # 可选：辅助 SupCon（默认关）
            if lambda_supcon > 0:
                loss_sup = supervised_contrastive_loss(pair_embs, labels, temperature=temperature)
                loss = loss_cls + lambda_supcon * loss_sup
            else:
                loss_known = known_only_supcon_loss(
                    pair_embs,
                    labels,
                    temperature=temperature
                )

                loss = loss_cls + 0.05 * loss_known

            optim.zero_grad()
            loss.backward()
            optim.step()

            total_loss += loss.item()
            steps += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg = total_loss / max(steps, 1)
        print(f"[Finetune] Epoch {ep} avg_loss={avg:.6f}")

        # ---- quick eval each epoch ----
        model.eval()
        y_true, y_pred = [], []
        with torch.no_grad():
            for batch in test_loader:
                logits, labels, _ = model(batch)
                y_true.append(labels.item())
                y_pred.append(logits.argmax(dim=1).item())

        report = classification_report(
            y_true, y_pred,
            target_names=["SLSF","SLDF","DLSF","DLDF"],
            digits=4,
            output_dict=True
        )
        f1_macro = report["macro avg"]["f1-score"]
        print(f"[Finetune] Epoch {ep} test_f1_macro={f1_macro:.4f}")

        if f1_macro > best_f1_macro:
            best_f1_macro = f1_macro
            torch.save({
                "encoder": model.encoder.state_dict(),
                "relation": model.relation.state_dict(),
                "classifier": model.classifier.state_dict(),
                "embedding_dim": embedding_dim,
                "num_layers": num_layers
            }, save_path)
            print(f"✨ Saved best finetuned model to {save_path} (best_f1_macro={best_f1_macro:.4f})")

    # final eval best
    print("\n===== Load best finetuned ckpt and evaluate =====")
    best = torch.load(save_path, map_location=device)
    model.encoder.load_state_dict(best["encoder"])
    model.relation.load_state_dict(best["relation"])
    model.classifier.load_state_dict(best["classifier"])
    evaluate_pair_model(model, test_loader, device, title="Final Test")

# =========================================================
# 9) 入口
# =========================================================
if __name__ == "__main__":
    # 你自己的路径
    train_jsonl = "generate_dataset/new_data/multiclass_dataset_train.jsonl"
    test_jsonl  = "generate_dataset/new_data/multiclass_dataset_test.jsonl"

    # ---------- Stage 1: SupCon (Relation) ----------
    # 关键：n_classes_per_batch * n_samples_per_class = batch size
    # 建议：4类任务，n_classes_per_batch=4，n_samples_per_class>=4
    train_supcon(
        train_jsonl=train_jsonl,
        embedding_dim=256,
        num_layers=4,
        n_classes_per_batch=4,
        n_samples_per_class=4,   # batch=16，每类4个 -> SupCon稳定
        lr=1e-3,
        epochs=50,
        temperature=0.1,
        patience=8,
        save_path="best_relation_supcon.pt",
    )

    # ---------- Stage 2: Finetune end-to-end ----------
    finetune(
        train_jsonl=train_jsonl,
        test_jsonl=test_jsonl,
        supcon_ckpt="best_relation_supcon.pt",
        embedding_dim=256,
        num_layers=4,
        batch_size=16,
        lr=1e-4,
        epochs=20,
        lambda_supcon=0.0,   # 先关；如果你发现 finetune 容易过拟合/崩，可以试 0.05
        temperature=0.1,
        save_path="best_relation_finetune.pt"
    )
