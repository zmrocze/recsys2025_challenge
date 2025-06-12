
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
      item_emb: (batch_size, emb_dim)
      returns: (batch_size, )
    """
    user_proj = torch.matmul(user_emb, self.Ws)  # (batch_size, hidden_dim)
    item_proj = torch.matmul(item_emb, self.Wt)  # (batch_size, hidden_dim)
    # compute similarity
    return torch.sum(user_proj * item_proj, dim=1)  # (batch_size, )


# gives number which correlates to the edge being there or not
class LinearEdgePredictor(torch.nn.Module):
  """
    Predicts the probability of an edge between user and item. Mimicks attention score calculation in GAT.
  """
  def __init__(self, embedding_dim, hidden_dim=None, dropout=0.0):
    super(LinearEdgePredictor, self).__init__()
    if hidden_dim is None:
      hidden_dim = embedding_dim
    self.Ws = torch.nn.Parameter(torch.Tensor(embedding_dim, hidden_dim)).to(device=device)
    self.Wt = torch.nn.Parameter(torch.Tensor(embedding_dim, hidden_dim)).to(device=device)
    self.relu = torch.nn.LeakyReLU(negative_slope=0.01)
    self.a = torch.nn.Parameter(torch.Tensor(1, hidden_dim)).to(device=device)
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

def l2_reg(self, alpha=0.01):
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
  def __init__(self, edge_index, trg_index_range, batch_size=256, neg_samples=1, random_state=None, device=device):
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
      neg_trg_node = self.target_index_range[0] + np.random.choice( 
          self.target_index_range[1] - self.target_index_range[0] ,
          size=(self.neg_samples * pos_indices.shape[0],), replace=True
        )
      
      yield src_node.to(device=self.device), \
            pos_trg_node.to(device=self.device), \
            int_tensor(neg_trg_node, device=self.device).view(-1, self.neg_samples)
  
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
  loss = - bpr_loss + l2_reg(gat.node_embeddings, alpha=l2_r) + l2_reg(gat, alpha=l2_r) + l2_reg(edge_predictor, alpha=l2_r)
  return loss

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

class BprTraining(pl.LightningModule):
  def __init__(self, recgat, edge_predictor, propensity_sku, lr=0.001, full_test_target=None, device=device, forward_gat_every_n=1):
    super(BprTraining, self).__init__()
    self.recgat = recgat.to(device=device)
    self._val_auroc_target = None
    self.edge_predictor = edge_predictor.to(device=device)
    self.propensity_sku = propensity_sku
    self.lr = lr
    self.full_test_target = full_test_target
    self._changed = True
    self._forward_skipped_n = 0
    self.forward_gat_every_n = forward_gat_every_n # 1 means recalculate every time. n>1 means recalculate after n backward passes

  # in principle whole epoch loss can be calculated after a single forward pass that updates the final layer embeddings
  # maybe could try it: but this risks not training well (would need low learning rate to keep stable (todo: test it))
  # on other hand recalculating every batch is costly
  # We recalculate every self.forward_gat_every_n.
  def get_final_layer_embeddings(self):
    # Recalculate every self.forward_gat_every_n backwards passes
    if self._changed and self._forward_skipped_n >= self.forward_gat_every_n:
      user_emb, item_emb = self.recgat.forward()
      self._user_emb = user_emb
      self._item_emb = item_emb
      self._changed = False
      self._forward_skipped_n = 0
    
    return self._user_emb, self._item_emb

  # calculate (based on val_loader) only first time its needed
  def get_val_auroc_target(self, user_id, item_id):
    if self._val_auroc_target is None:
      edge_index = torch.stack((user_id, item_id), device=device)
      self._val_auroc_target = create_target_from_edge_index(self.recgat.node_id_map, self.recgat.node_id_map.n_users, self.propensity_sku, edge_index)
    return self._val_auroc_target
    

  def optimizer_step(self, *args, **kwargs):
    self._changed = True # record that gat has to be recomputed
    self._forward_skipped_n += 1
    return super().optimizer_step(*args, **kwargs)

  def training_step(self, batch, batch_idx):
    src_node, pos_trg_node, neg_trg_node = batch
    pos_scores = self.forward(src_node, pos_trg_node).view(-1, 1)
    neg_scores = self.forward(src_node.view(-1, 1), neg_trg_node)
    loss = loss_f(self.recgat, self.edge_predictor, pos_scores, neg_scores)

    self.log('train_loss', loss)
    return loss

  def configure_optimizers(self):
    return torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=0.001)
  
  def validation_step(self, batch, batch_idx):
    src_node, pos_trg_node, neg_trg_node = batch
    pos_scores = self.forward(src_node, pos_trg_node).view(-1, 1)
    neg_scores = self.forward(src_node.view(-1, 1), neg_trg_node)
    loss = loss_f(self.recgat, self.edge_predictor, pos_scores, neg_scores)

    # auroc = self.auroc_on_propensity(self.get_val_auroc_target(src_node, pos_trg_node)) # just pos_trg_node !
    
    if self.full_test_target is not None:
      full_auroc = self.auroc_on_propensity(self.full_test_target)
      self.log("val_propensity_auroc", full_auroc)

    self.log("val_loss", loss)
    return loss

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

  # Auroc for users (defaul all) on given items (default all), given binary target from test (say 14 days into future)
  def auroc_on_propensity(self, target_edge_index):
    propensity_item_id = torch.tensor([ self.recgat.node_id_map.id_of_item[sku] for sku in self.propensity_sku ], dtype=torch.long, device=device)
    user_id = torch.arange(0, self.recgat.node_id_map.n_users, dtype=torch.long, device=device)  # all users
    # repeat calculation from validation_step but for smaller item set so ignoring
    scores = self.forward(user_id.view(-1, 1), propensity_item_id)
    return tm.AUROC(task="binary")(scores, target_edge_index)
    # return torch.zeros((1,), device=device)  # TODO: implement AUROC calculation