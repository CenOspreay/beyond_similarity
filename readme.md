## Project Structure

```text
├── build_uCFG/                # Scripts for building uncontextualized Context-Free Grammars (uCFG)
├── dataset/                   # Directory for storing training and evaluation datasets
├── model.py                   # Core definition of the CenOsprey model architecture
├── train_relation_supcon_finetune.py # Training script for Relation Extraction with SupCon fine-tuning
├── train_3class.py            # Training script for standard 3-class classification tasks
└── test_3class.py             # Evaluation script for testing 3-class classification models
```

##  Quick Start

### 1. Data Preparation

Dataset files in the `dataset/` directory.

### 2. Training

**Option A: Relation Extraction (SupCon Fine-tuning)**
To train the model for relation extraction using Supervised Contrastive Learning:

```bash
python train_relation_supcon_finetune.py
```

**Option B: Standard 3-Class Classification**
To train a baseline 3-class classifier:

```bash
python train_3class.py
```

### 3. Evaluation

To evaluate a trained 3-class model checkpoint:

```bash
python test_3class.py
```

