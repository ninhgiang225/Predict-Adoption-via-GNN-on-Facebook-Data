# Predict-Adoption-via-GNN-on-Facebook-Data
This project aims to predict whether a user in a social network will adopt a new technology, based on their network position and behavioral attributes. The task is modeled as a **node classification** problem using **graph neural networks (GNNs)**.

## Phase 1: Define the Problem

### Problem Type
- **Type**: Supervised learning – Node classification
- **Prediction Goal**: For each user (node), predict a binary label:
  - `1` → Will adopt the technology
  - `0` → Will not adopt

---

### Input: Node Features
Each node (user) is represented with a feature vector composed of:

| Feature               | Description                                                                 |
|------------------------|-----------------------------------------------------------------------------|
| `degree`              | Number of direct neighbors (friends)                                        |
| `activity_frequency`  | Frequency of user interactions or simulated activity level                  |
| *(Optional)* `influence_score` | Fraction of neighbors who have adopted (can be used in advanced models)    |

These features reflect a user's position and behavior within the network.

---

### Input: Graph Structure
The graph structure is derived from the **Facebook Social Circles** dataset, where:
- Nodes represent users
- Edges represent friendship connections

This social graph provides the relational structure necessary for GNNs to learn peer influence.

---

### Output: Adoption Label
Since the dataset does not include real-world adoption data, we simulate it as follows:

1. **Seed Adoption**: Randomly assign `10%` of users as early adopters (label = 1).
2. **Remaining Users**: Assigned label = 0 (not adopted).
3. *(Optional)* In later phases, simulate influence propagation using threshold or cascade models.

---
## Phase 2: Dataset Preparation

### Dataset Source
We use the **Facebook Social Circles** dataset from the [Stanford SNAP collection](https://snap.stanford.edu/data/ego-Facebook.html), specifically the `facebook_combined.txt` file, which contains:
- 4,039 nodes (users)
- 88,234 undirected edges (friendship links)

---

### Graph Loading

We load the data as an undirected graph using NetworkX:

```python
G = nx.read_edgelist("facebook_combined.txt", create_using=nx.Graph(), nodetype=int)
```

## Phase 4: Define the Problem Methodology

### Model
- **Architecture:** 2-layer Graph Convolutional Network (GCN) + Global Signal Extension  
- **Framework:** [PyTorch Geometric (PyG)](https://pytorch-geometric.readthedocs.io/)  
- **Loss:** Cross-Entropy Loss  
- **Optimizer:** Adam  
- **Training Details:**  
  - Learning rate: `0.03`  
  - Weight decay: `1e-4`  
  - Epochs: ~200 (early stopping used)  
  - Batch loading: `NeighborLoader`

### Training Process
1. Load dataset & preprocess features  
2. Create train/validation/test splits  
3. Train GNN model with mini-batch sampling  
4. Evaluate accuracy & visualize predictions  

---

## Phase 5: Results

### Accuracy
- **Baseline GCN:** ~80%  
- **GCN with Global Signal:** 72% (accuracy drop observed)  

### Confusion Matrix (Global Signal)
|               | Predicted 0 | Predicted 1 |
|---------------|-------------|-------------|
| **True 0**    | High        | 119         |
| **True 1**    | 1015        | High        |

**Observation:**
- The model predicts **non-adopters** far more often  
- Global trends in visualization match real adoption spread  
- Fine-grained classification suffers (especially for adopters)  

## Visualization
- Graph plot showing predicted adoption clusters  
- Color-coded by predicted vs true label  
- Reveals general trend alignment but **underprediction** for adopters  

## Limitations
- Simulated labels may not reflect real adoption patterns  
- Model underpredicts adoption cases in the extended GCN version  
- Features are basic — richer attributes (e.g., age, activity history, communities) could improve results  

---

## Phase 6: Extension
Integrate Global Graph Properties in PyG implementation to solve the restrict of local passing
