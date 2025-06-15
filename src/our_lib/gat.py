
import numpy as np
import torch
import torch_geometric as tg
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import torchmetrics as tm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class NodeIdMap:
  def __init__(self, users, items):
    self.id_of_user = {user: idx for idx, user in enumerate(users)}
    self.id_of_item = {item: idx + len(users) for idx, item in enumerate(items)}
    self.user_of_id = {idx: user for user, idx in self.id_of_user.items()}
    self.item_of_id = {idx: item for item, idx in self.id_of_item.items()}
    self.n_users = len(users)
    self.n_items = len(items)
    self.N = self.n_users + self.n_items

  def make_edges(self, users, items):
    user_ids = np.array([self.id_of_user[u] for u in users], dtype=np.long)
    item_ids = np.array([self.id_of_item[i] for i in items], dtype=np.long)
    return torch.tensor(np.stack((user_ids, item_ids), axis=0), dtype=torch.long, device=device)


embedding_dim = 128
dropout=0.2
# heads = 8
heads = 1 # in the end maybe better to increase dimension than add more heads?

class JustGAT(torch.nn.Module):

  def __init__(self, n, embedding_dim=embedding_dim, edge_dim=None, device=device, heads=heads, dropout=0.2, a=1.0, type='normal', num_layers=2):
    super(JustGAT, self).__init__()
    self.embedding_dim = embedding_dim
    self.edge_dim = edge_dim
    self.n = n
    self.node_embeddings = torch.nn.Embedding(self.n, embedding_dim, device=device)
    # self.node_embeddings.weight.data.uniform_(-0.01, 0.01)
    # self.target_embeddings = torch.nn.Embedding(M, embedding_dim, device=device)
    # v2=True, 
    # heads, concat, residual
    self.gat = tg.nn.models.GAT(
      in_channels=self.embedding_dim, 
      hidden_channels=self.embedding_dim, 
      out_channels=None,
      num_layers=num_layers,
      v2=True,
      dropout=dropout,
      # GATv2Conv
      heads=heads,
      residual=True,
      edge_dim=self.edge_dim,
      # concat=
    ).to(device)
    self.a = a
    self.type = type
    self.reinit_weights(a=self.a, type=self.type)

  def reinit_weights(self, a=1.0, type='normal'):
    if type == 'normal':
      self.node_embeddings.weight.data.normal_(0, a)
    elif type == 'uniform':
      self.node_embeddings.weight.data.uniform_(-a, a)
    else:
      raise ValueError(f"Unknown weight initialization type: {type}. Use 'normal' or 'uniform'.")
    self.gat.reset_parameters()

  def forward(self, edge_index, edge_weight=None, edge_attr=None):
    return self.gat.forward(
      x=self.node_embeddings.weight, 
      edge_index=edge_index, 
      edge_weight=edge_weight,
      edge_attr=edge_attr
    )

class JustLightGCN(torch.nn.Module):

  def __init__(self, n, embedding_dim=embedding_dim, device=device, a=1.0, type='normal', num_layers=2):
    super(JustLightGCN, self).__init__()
    self.embedding_dim = embedding_dim
    self.n = n
    self.node_embeddings = torch.nn.Embedding(self.n, embedding_dim, device=device)
    # self.node_embeddings.weight.data.uniform_(-0.01, 0.01)
    # self.target_embeddings = torch.nn.Embedding(M, embedding_dim, device=device)
    # v2=True, 
    # heads, concat, residual
    self.gat = tg.nn.models.LightGCN(
      num_nodes=self.n,
      embedding_dim=self.embedding_dim,
      num_layers=num_layers,
      ).to(device)
    self.a = a
    self.type = type
    self.reinit_weights(a=self.a, type=self.type)

  def reinit_weights(self, a=1.0, type='normal'):
    if type == 'normal':
      self.node_embeddings.weight.data.normal_(0, a)
    elif type == 'uniform':
      self.node_embeddings.weight.data.uniform_(-a, a)
    else:
      raise ValueError(f"Unknown weight initialization type: {type}. Use 'normal' or 'uniform'.")
    self.gat.reset_parameters()

  def forward(self, edge_index, edge_weight=None, edge_attr=None):
    return self.gat.forward(
      x=self.node_embeddings.weight, 
      edge_index=edge_index, 
      edge_weight=edge_weight,
      edge_attr=edge_attr
    )


def test_out():
  # Initialize the model
  model = JustGAT(10, embedding_dim=16, edge_dim=3)

  edge_index = torch.tensor([[1, 1, 1, 2, 3], [0, 1, 2, 1, 3]], dtype=torch.long, device=device)
  edge_weight = torch.ones(5, dtype=torch.float, device=device)
  edge_attr = torch.ones(5, 3, dtype=torch.float, device=device)  # 3 edge types
  print(edge_attr.shape)
  # Forward pass
  with torch.no_grad():
    output = model.forward(edge_index, edge_weight=edge_weight, edge_attr=edge_attr) #  edge_attr=edge_attr
    print(f"Output shapes: {output.shape}")

class RecLightGCN(JustLightGCN):
  """
    Wrapper over JustLightGCN that creates the graph from recommendation data.
  """
  def __init__(self, users, items, **kwargs):
    self.node_id_map = NodeIdMap(users, items)
    
    JustLightGCN.__init__(self, self.node_id_map.N, **kwargs)

    # self.gat = JustLightGCN(self.node_id_map.N, **kwargs).to(device)
    self.edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
    # self.edge_attr = torch.empty((0, self.model.edge_dim), dtype=torch.float, device=device)
    self.edge_attr = None
    # self.edge_weights = torch.empty((0, ), dtype=torch.float, device=device)
    self.edge_weight = None

  def add_edges(self, users, items, edge_attr=None, edge_weight=None):
    edge_index = self.node_id_map.make_edges(users, items).to(device=device)
    edge_index = torch.concat((self.edge_index, edge_index), dim=1)
    return self.add_edge_index(edge_index, edge_attr=edge_attr, edge_weight=edge_weight)

  def add_edge_index(self, edge_index, edge_attr=None, edge_weight=None):
    self.edge_index = torch.concat((self.edge_index, edge_index), dim=1)
    if edge_attr is not None:
      assert self.edge_dim is not None
      self._init_edge_attr()
      self.edge_attr = torch.concat((self.edge_attr, edge_attr))
    else:
      assert self.edge_dim is None
    if edge_weight is not None:
      self._init_edge_weight()
      self.edge_weight = torch.concat((self.edge_weight, edge_weight))
    self._assert_edge_assignment()

  def _init_edge_attr(self):
    if self.edge_attr is None: self.edge_attr = torch.empty((0, self.edge_dim), dtype=torch.float, device=device)
  def _init_edge_weight(self):
    if self.edge_weight is None: self.edge_weight = torch.empty((0, ), dtype=torch.float, device=device)
  def _assert_edge_assignment(self):
    if (self.edge_attr is not None and self.edge_index.shape[1] != self.edge_attr.shape[0]) or (self.edge_weight is not None and self.edge_index.shape[1] != self.edge_weight.shape[0]):
      raise ValueError(f"Edge index, edge attr and edge weight must have the same number of edges.")

  # assumes: df['client_id'] and df['sku'] are present
  def add_edges_from_user_item_df(self, df, edge_attr=None, edge_weight=None):
    user_ids = df.client_id.values
    item_ids = df.sku.values
    self.add_edges(user_ids, item_ids, edge_attr=edge_attr, edge_weight=edge_weight)
  
  def add_edges_from_user_category_df(self, df, edge_attr=None, edge_weight=None):
    user_ids = df.client_id.values
    item_ids = df.category.values
    self.add_edges(user_ids, item_ids, edge_attr=edge_attr, edge_weight=edge_weight)
    
  def forward(self):
    y = JustLightGCN.forward(self, edge_index=self.edge_index, edge_weight=self.edge_weight, edge_attr=self.edge_attr)
    # split into user and item embeddings
    user_embeddings = y[:self.node_id_map.n_users]
    item_embeddings = y[self.node_id_map.n_users:]
    return user_embeddings, item_embeddings


class RecGAT(JustGAT):
  """
    Wrapper over JustGAT that creates the graph from recommendation data.
  """
  def __init__(self, users, items, **kwargs):
    self.node_id_map = NodeIdMap(users, items)
    
    JustGAT.__init__(self, self.node_id_map.N, **kwargs)

    # self.gat = JustGAT(self.node_id_map.N, **kwargs).to(device)
    self.edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
    # self.edge_attr = torch.empty((0, self.model.edge_dim), dtype=torch.float, device=device)
    self.edge_attr = None
    # self.edge_weights = torch.empty((0, ), dtype=torch.float, device=device)
    self.edge_weight = None

  def add_edges(self, users, items, edge_attr=None, edge_weight=None):
    edge_index = self.node_id_map.make_edges(users, items).to(device=device)
    # edge_index = torch.concat((self.edge_index, edge_index), dim=1)
    return self.add_edge_index(edge_index, edge_attr=edge_attr, edge_weight=edge_weight)

  def add_edge_index(self, edge_index, edge_attr=None, edge_weight=None):
    self.edge_index = torch.concat((self.edge_index, edge_index), dim=1)
    if edge_attr is not None:
      assert self.edge_dim is not None
      self._init_edge_attr()
      self.edge_attr = torch.concat((self.edge_attr, edge_attr))
    else:
      assert self.edge_dim is None
    if edge_weight is not None:
      self._init_edge_weight()
      self.edge_weight = torch.concat((self.edge_weight, edge_weight))
    self._assert_edge_assignment()

  def _init_edge_attr(self):
    if self.edge_attr is None: self.edge_attr = torch.empty((0, self.edge_dim), dtype=torch.float, device=device)
  def _init_edge_weight(self):
    if self.edge_weight is None: self.edge_weight = torch.empty((0, ), dtype=torch.float, device=device)
  def _assert_edge_assignment(self):
    if (self.edge_attr is not None and self.edge_index.shape[1] != self.edge_attr.shape[0]) or (self.edge_weight is not None and self.edge_index.shape[1] != self.edge_weight.shape[0]):
      raise ValueError(f"Edge index, edge attr and edge weight must have the same number of edges.")

  # assumes: df['client_id'] and df['sku'] are present
  def add_edges_from_user_item_df(self, df, edge_attr=None, edge_weight=None):
    user_ids = df.client_id.values
    item_ids = df.sku.values
    self.add_edges(user_ids, item_ids, edge_attr=edge_attr, edge_weight=edge_weight)
  
  def add_edges_from_user_category_df(self, df, edge_attr=None, edge_weight=None):
    user_ids = df.client_id.values
    item_ids = df.category.values
    self.add_edges(user_ids, item_ids, edge_attr=edge_attr, edge_weight=edge_weight)
    
  def forward(self):
    y = JustGAT.forward(self, edge_index=self.edge_index, edge_weight=self.edge_weight, edge_attr=self.edge_attr)
    # split into user and item embeddings
    user_embeddings = y[:self.node_id_map.n_users]
    item_embeddings = y[self.node_id_map.n_users:]
    return user_embeddings, item_embeddings


# users = data.add_to_cart.client_id.unique()
# items = data.add_to_cart.sku.unique()
# model = RecGAT(users, items, embedding_dim=8, edge_dim=None)
# model.add_edges_from_user_item_df(data.add_to_cart)
# us, it = model.forward()
# us.shape, it.shape


# assumes edges have one of n types, encoded in edge_attr as one-hot vectors, predicts probability of each
# !! cant? because there can be more than one type of edge

# gives number which correlates to the edge being there or not
# Untested yet!
class DotproductEdgePredictor(torch.nn.Module):
  """
    Untested yet!!!
    Predicts the probability of an edge between user and item.
  """
  def __init__(self, emb_dim):
    super(DotproductEdgePredictor, self).__init__()
    self.Ws = torch.nn.Parameter(torch.Tensor(emb_dim, emb_dim)).to(device=device)
    self.Wt = torch.nn.Parameter(torch.Tensor(emb_dim, emb_dim)).to(device=device)
    
  def forward(self, user_emb, item_emb):
    """
      user_emb: (batch_size, emb_dim)
      item_emb: (batch_size, sample_size, emb_dim)
      returns: (batch_size, )
    """
    user_proj = torch.matmul(user_emb, self.Ws)  # (batch_size, hidden_dim)
    item_proj = torch.matmul(item_emb, self.Wt)  # (batch_size, sample_size, hidden_dim)
    # compute similarity
    return torch.sum(user_proj * item_proj, axis=len(item_proj.shape)-1)  # (batch_size, )


# gives number which correlates to the edge being there or not
class LinearEdgePredictor(torch.nn.Module):
  """
    Predicts the probability of an edge between user and item. Mimicks attention score calculation in GAT.
  """
  def __init__(self, embedding_dim, hidden_dim=None, dropout=0.0):
    super(LinearEdgePredictor, self).__init__()
    if hidden_dim is None:
      hidden_dim = embedding_dim
    self.Ws = torch.nn.Parameter(torch.Tensor(embedding_dim, hidden_dim))
    self.Wt = torch.nn.Parameter(torch.Tensor(embedding_dim, hidden_dim))
    self.relu = torch.nn.LeakyReLU(negative_slope=0.01)
    self.a = torch.nn.Parameter(torch.Tensor(1, hidden_dim))
    self.dropout = torch.nn.Dropout(p=dropout)
    self.init_weights()

  def init_weights(self, a = 1.0, type='normal'):
    if type == 'normal':
      self.a.data.normal_(0, a)
      self.Ws.data.normal_(0, a)
      self.Wt.data.normal_(0, a)
    elif type == 'uniform':
      range = (a * -1.0, a * 1.0)
      self.a.data.uniform_(*range)
      self.Ws.data.uniform_(*range)
      self.Wt.data.uniform_(*range)
    else:
      raise ValueError(f"Unknown weight initialization type: {type}. Use 'normal' or 'uniform'.")

  def forward(self, user_emb, item_emb):
    """
      user_emb: (batch_size, emb_dim)
      item_emb: (batch_size, emb_dim) OR (batch_size, sample_size, emb_dim)
      returns: (batch_size, )
    """
    user_proj = torch.matmul(user_emb, self.Ws)  # (batch_size, hidden_dim)
    item_proj = torch.matmul(item_emb, self.Wt)  # (batch_size, hidden_dim)
    # compute similarity
    # print(self.dropout(self.relu(user_proj + item_proj)).shape)
    # print(self.a.shape)
    # print((torch.mm(self.dropout(self.relu(user_proj + item_proj)), self.a)).shape)

    return  torch.sum( self.a * self.dropout(self.relu(user_proj + item_proj)), axis=len(item_emb.shape)-1)   # (batch_size, 1)

def l2_reg(self):
  i=0
  total = torch.zeros((1,), device=device)
  for p in self.parameters():
    total += (p ** 2).mean()
    i += 1
  total /= i
  return total

# TODO: figure out how to not leak negative edges into test
#       Do we gotta use these torch geometric things?
def train_test_split_pos_edges(edge_index, test_size=0.2, random_state=None):
  if random_state is not None:
    np.random.seed(random_state)
  
  num_edges = edge_index.shape[1]
  num_test = int(num_edges * test_size)

  indices = np.random.permutation(num_edges)
  test_indices = indices[:num_test]
  train_indices = indices[num_test:]

  return edge_index[:, train_indices], edge_index[:, test_indices]

def int_tensor(np_array, device):
  return torch.tensor(np_array, dtype=torch.long, device=device)

class BprLossLoader:
  """
    Loads batches of positive and negative edges (just random) for BPR loss.
    Uses edge_index to sample negative edges.
  """
  def __init__(self, edge_index, trg_index_range, batch_size=256, neg_samples=1, random_state=None, device=device, sample_neg_from_all=False):
    self.sample_neg_from_all = sample_neg_from_all
    self.edge_index = edge_index
    self.batch_size = batch_size
    self.neg_samples = neg_samples
    self.random_state = random_state
    self.device = device
    self.num_edges = edge_index.shape[1]
    self.target_index_range = trg_index_range  # range of indices to sample target endpoints for negative (nonexisting) edges
    self.indices = np.arange(self.num_edges)
  
  def __iter__(self):
    if self.random_state is not None:
      np.random.seed(self.random_state)
    
    np.random.shuffle(self.indices)
    
    for start in range(0, self.num_edges, self.batch_size):
      end = min(start + self.batch_size, self.num_edges)
      pos_indices = self.indices[start:end]
      # pos_edges = self.edge_index[:, pos_indices]
      src_node = self.edge_index[0, pos_indices]
      pos_trg_node = self.edge_index[1, pos_indices]
      if self.sample_neg_from_all:
        neg_trg_node = self.target_index_range[0] + np.random.choice( 
            self.target_index_range[1] - self.target_index_range[0] ,
            size=(self.neg_samples * pos_indices.shape[0],), replace=True
          )
        neg_trg_node = int_tensor(neg_trg_node, device=self.device)
      else:
        neg_trg_node_ind = np.random.choice(np.arange(0, pos_trg_node.shape[0]), size=(self.neg_samples * pos_indices.shape[0],), replace=True)
        neg_trg_node = pos_trg_node[neg_trg_node_ind]
      
      yield src_node.to(device=self.device), \
            pos_trg_node.to(device=self.device), \
            neg_trg_node.view(-1, self.neg_samples)
  
  def __len__(self):
    return (self.num_edges + self.batch_size - 1) // self.batch_size  # ceil division
  
def unique_edges(df):
  """
    Returns a DataFrame with unique edges from the given DataFrame.
    Assumes df has 'client_id' and 'sku' columns.
  """
  return df.drop_duplicates(subset=['client_id', 'sku']).reset_index(drop=True)


def loss_f(gat, edge_predictor, pos_scores, neg_scores, l2_r=0.01):
  bpr_loss = torch.nn.functional.logsigmoid(pos_scores - neg_scores).mean()
  loss = - bpr_loss + l2_r * (l2_reg(gat.node_embeddings) + l2_reg(gat) + l2_reg(edge_predictor))
  return loss

def bpr_loss_f(pos_scores, neg_scores):
  bpr_loss = torch.nn.functional.logsigmoid(pos_scores - neg_scores).mean()
  loss = - bpr_loss
  return loss

def loss_ff(gat, edge_predictor, pos_scores, neg_scores, l2_r=0.01):
  bpr_loss = torch.nn.functional.logsigmoid(pos_scores - neg_scores).mean()
  # loss =  + 
  return (- bpr_loss), l2_r * (l2_reg(gat.node_embeddings) + l2_reg(gat) + l2_reg(edge_predictor))


# to calculate AUROC we need a matrix (users, 100) of 1s and 0s whether edge is or isnt
# we have list of edges. better do this once, not vectorized
def create_target_from_edge_index(node_id_map, n_users, propensity_items, edge_index):
  ind = { sku : i for i, sku in enumerate(propensity_items) }
  target = torch.zeros((n_users, len(ind)), dtype=torch.int, device=device) # (n_users, 100)
  for i in range(edge_index.shape[1]):
    user_id = edge_index[0, i].item()
    item = node_id_map.item_of_id[edge_index[1, i].item()]
    if item in ind:
      target[user_id, ind[item]] = 1
  return target.to(device=device)

# to calculate AUROC we need a matrix (users, 100) of 1s and 0s whether edge is or isnt
# we have list of edges. better do this once, not vectorized
# all tensors with node id
def create_batch_assoc_matrix(edge_index, users=None, items=None, device=device):
  if users is None:
    users = edge_index[0, :].unique()
  if items is None:
    items = edge_index[1, :].unique()

  a = torch.zeros((users.shape[0], items.shape[0]), dtype=torch.int).to(device=device)

  user_ind = {user.item(): i for i, user in enumerate(users)}
  item_ind = {item.item(): i for i, item in enumerate(items)}

  for i in range(edge_index.shape[1]):
    user = edge_index[0, i].item()
    item = edge_index[1, i].item()
    if user in user_ind and item in item_ind:
      a[user_ind[user], item_ind[item]] = 1

  return a.to(device=device)

import torchmetrics as tm

# from torcheval.metrics import BinaryAUROC

def create_val_edge_batched(node_id_map, val_edge_index, auroc_batch_size, device=device):
  num_batches = int(node_id_map.n_users // auroc_batch_size + (node_id_map.n_users % auroc_batch_size > 0))
  val_edge_index_batched = [None for _ in range(num_batches)]
  val_users = val_edge_index[0, :].unique()
  n_users = val_users.shape[0] 
  # the point is to split val_edge_index into (uneven) batches of edges corresponding to batched user ids
  for i, start in enumerate(range(0, n_users, auroc_batch_size)):
    # assuming user id's go from 0 to n_users-1, then item ids
    end = min(start + auroc_batch_size, node_id_map.n_users)
    # batch_users = torch.arange(start, end, dtype=torch.long)
    batch_users = {u.item(): i for i, u in enumerate(val_users[start:end])}
    # ind = torch.nonzero((start <= val_edge_index[0, :]) & (val_edge_index[0, :] < end) , as_tuple=False).to(device=device)
    # val_edge_index_batched[i] = val_edge_index[:, ind].squeeze(2).to(device=device)
    batch_target = torch.zeros((start-end, node_id_map.n_items), dtype=torch.long, device=device)
    for j in range(val_edge_index.shape[1]):
      user = val_edge_index[0, j].item()
      item = val_edge_index[1, j].item()
      if user in batch_users:
        item_ind = item - node_id_map.n_users
        batch_target[batch_users[user], item_ind] = 1
    val_edge_index_batched[i] = batch_target.to(device=device)
  
  return val_edge_index_batched

def create_val_target_batched(node_id_map, val_edge_index, auroc_batch_size, device=device):
  raise Exception("What about the fact that items have indices going n_users..N-1?")
  num_batches = int(node_id_map.n_users // auroc_batch_size + (node_id_map.n_users % auroc_batch_size > 0))
  val_edge_index_batched = [None for _ in range(num_batches)]
  # the point is to split val_edge_index into (uneven) batches of edges corresponding to batched user ids
  for i, start in enumerate(range(0, node_id_map.n_users, auroc_batch_size)):
    # assuming user id's go from 0 to n_users-1, then item ids
    end = min(start + auroc_batch_size, node_id_map.n_users)
    # batch_users = torch.arange(start, end, dtype=torch.long)
    ind = torch.nonzero((start <= val_edge_index[0, :]) & (val_edge_index[0, :] < end) , as_tuple=False).to(device=device)
    target_edges = val_edge_index[:, ind].squeeze(2).to(device=device)
    # remapping users to 0..(start-end)
    target_edges[0, :] = target_edges[0, :] - start
    target = torch.sparse_coo_tensor(
      target_edges, 
      torch.ones((target_edges.shape[1],)), 
      size=(auroc_batch_size, node_id_map.n_items), 
      dtype=torch.long, device=device
    )
    val_edge_index_batched[i] = target

  return val_edge_index_batched

  


class BprTraining(pl.LightningModule):
  def __init__(self, recgat, edge_predictor, retain_grad=False,
    lr=0.001, l2_reg=0.01, val_edge_index=None, device=device, auroc_batch_size=256,
    # forward_gat_every_n=1, 
    val_edge_index_batched=None,
    patience=5, factor=0.5, lr_scheduler_monitor="train_loss"):
    super(BprTraining, self).__init__()
    self.retain_grad = retain_grad
    self.patience = patience
    self.factor = factor
    self.lr_scheduler_monitor = lr_scheduler_monitor
    self.recgat = recgat.to(device=device)
    self.edge_predictor = edge_predictor.to(device=device)
    # self.propensity_sku = propensity_sku
    self.lr = lr
    # self.full_test_target = full_test_target
    self._changed = True
    # self._forward_skipped_n = 0
    # self._first_forward = True
    self.l2_reg = l2_reg
    # self.forward_gat_every_n = forward_gat_every_n # 1 means recalculate every time. n>1 means recalculate after n backward passes
    self.auroc_batch_size = auroc_batch_size
    self.val_edge_index_batched = val_edge_index_batched
    self.val_users = val_edge_index[0, :].unique() # has to match create_val_edge_batched
    # "edges "sorted" in order of batches of val_users"
    if val_edge_index is not None and self.val_edge_index_batched is None:
      self.val_edge_index_batched = create_val_edge_batched(self.recgat.node_id_map, val_edge_index, self.auroc_batch_size, device=device)
    # self.val_target_batched = create_val_target_batched(self.recgat.node_id_map, val_edge_index, self.auroc_batch_size, device=self.device) if val_edge_index is not None else None

  def on_save_checkpoint(self, checkpoint):
    checkpoint['my_node_id_map'] = self.recgat.node_id_map
    user_emb, item_emb = self.get_final_layer_embeddings()
    checkpoint['my_user_emb'] = user_emb
    checkpoint['my_item_emb'] = item_emb

  # in principle whole epoch loss can be calculated after a single forward pass that updates the final layer embeddings
  # maybe could try it: but this risks not training well (would need low learning rate to keep stable (todo: test it))
  # on other hand recalculating every batch is costly
  # We recalculate every self.forward_gat_every_n.
  # ! 2 forwards per epoch (pos and neg)
  def get_final_layer_embeddings(self):
    # Recalculate every self.forward_gat_every_n backwards passes
    if self._changed:
      user_emb, item_emb = self.recgat.forward()
      self._user_emb = user_emb
      self._item_emb = item_emb
      self._changed = False

    return self._user_emb, self._item_emb

  def training_step(self, batch, batch_idx):
    src_node, pos_trg_node, neg_trg_node = batch
    pos_scores = self.forward(src_node, pos_trg_node).view(-1, 1)
    neg_scores = self.forward(src_node.view(-1, 1), neg_trg_node)
    loss = bpr_loss_f(pos_scores, neg_scores)
    self.log('train_loss', loss)

    return loss

  def optimizer_step(
        self,
        epoch,
        batch_idx,
        optimizer,
        optimizer_closure,
    ) -> None:
    # default
    optimizer.step(closure=optimizer_closure)
    # custom
    self._changed = True

  # def optimizer_zero_grad(self, epoch, batch_idx, optimizer) -> None:
  #   # default
  #   optimizer.zero_grad()

  def backward(self, loss):
    loss.backward(retain_graph=self.retain_grad)

  def configure_optimizers(self):
    optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.l2_reg)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
      optimizer=optimizer,
      patience=self.patience,
      factor=self.factor,
      cooldown=2,
      verbose=True,
    )
    return {
      # ! be worry: l2 loss applies to all parameters every optimizer epoch. but bpr_loss touches only some users and items
      "optimizer": optimizer,
      "lr_scheduler": {
        "scheduler" : lr_scheduler,
        "monitor": self.lr_scheduler_monitor,
        "interval": "epoch",
        "frequency": 1,
      }
    }
  
  def validation_step(self, batch, batch_idx):
    src_node, pos_trg_node, neg_trg_node = batch
    pos_scores = self.forward(src_node, pos_trg_node).view(-1, 1)
    neg_scores = self.forward(src_node.view(-1, 1), neg_trg_node)
    # loss = loss_f(self.recgat, self.edge_predictor, pos_scores, neg_scores, l2_r=self.l2_reg)
    # bpr_loss = loss_f(self.recgat, self.edge_predictor, pos_scores, neg_scores)
    loss = bpr_loss_f(pos_scores, neg_scores)

    self.log("val_loss", loss)
    return loss

  def on_validation_epoch_end(self):
    if self.val_edge_index_batched is not None:
      full_auroc = self.auroc()
      self.log("val_auroc", full_auroc)

    super(BprTraining, self).on_validation_epoch_end()

  # helpers

  def reinit_weights(self, a=1.0, type='normal'):
    self.recgat.reinit_weights(a=a, type=type)
    self.edge_predictor.init_weights(a=a, type=type)
  
  # returns scores
  # item_id : node id of item node (therefore "- n_users")
  def forward(self, user_id, item_id):
    user_emb, item_emb = self.get_final_layer_embeddings()
    batch_user_emb = user_emb[user_id]
    batch_item_emb = item_emb[item_id - self.recgat.node_id_map.n_users]
    scores = self.edge_predictor(batch_user_emb, batch_item_emb)
    return scores  # (batch_size, item_id.shape[1])

  # metrics: 

  # averaged over users.
  # Corresponds to probability that a randomly selected user is predicted to buy item he actually bought over one he didn't.
  def auroc_og(self):
    # all_val_users, all_categories, val_edge_index_batched
    all_categories = torch.arange(self.recgat.node_id_map.n_users, self.recgat.node_id_map.N, dtype=torch.long, device=self.device)
    n_users = self.recgat.node_id_map.n_users
    auroc_acc = tm.AUROC(task="binary")

    for i, start in enumerate(range(0, n_users, self.auroc_batch_size)):
      # assuming user id's go from 0 to n_users-1, then item ids
      end = min(start + self.auroc_batch_size, n_users)
      batch_users = torch.arange(start, end, dtype=torch.long, device=self.device)
      batch_edges = self.val_edge_index_batched[i] # i-th batch, !! careful
      # create target matrix for this batch
      target = create_batch_assoc_matrix(batch_edges, users=batch_users, items=all_categories, device=self.device)
      # calculate scores
      scores = self.forward(batch_users.view(-1, 1), all_categories.view(1, -1))
      assert target.shape == scores.shape, f"Target shape {target.shape} does not match scores shape {scores.shape}"
      print(f"Batch {i}: target shape {target.device}, scores shape {scores.device}")
      # calculate AUROC
      # auroc_acc.update(scores, target)
      roc = auroc_acc(scores, target)
      roc_total += roc

    roc_total /= n_users
    return roc_total

# averaged over users.
  # Corresponds to probability that a randomly selected user is predicted to buy item he actually bought over one he didn't.
  def auroc(self):
    # all_val_users, all_categories, val_edge_index_batched
    all_categories = torch.arange(self.recgat.node_id_map.n_users, self.recgat.node_id_map.N, dtype=torch.long, device=self.device)
    n_users = self.recgat.node_id_map.n_users
    auroc_acc = tm.AUROC(task="binary")

    for i, start in enumerate(range(0, n_users, self.auroc_batch_size)):
      # assuming user id's go from 0 to n_users-1, then item ids
      end = min(start + self.auroc_batch_size, n_users)
      batch_users = torch.arange(start, end, dtype=torch.long, device=self.device)
      # batch_edges = self.val_edge_index_batched[i] # i-th batch, !! careful
      # # create target matrix for this batch
      # target = create_batch_assoc_matrix(batch_edges, users=batch_users, items=all_categories, device=self.device)
      batch_target_coo = self.val_target_batched[i]
      # calculate scores
      scores = self.forward(batch_users.view(-1, 1), all_categories.view(1, -1))
      assert batch_target_coo.shape == scores.shape, f"Target shape {batch_target_coo.shape} does not match scores shape {scores.shape}"
      # calculate AUROC
      # auroc_acc.update(scores, target)
      roc = auroc_acc(scores, batch_target_coo)
      roc_total += roc

    roc_total /= n_users
    return roc_total
