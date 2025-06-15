
from recsys_cmds import validate_and_load_embeddings # function
from recsys_cmds import split_data, create_baseline_embeddings, contest_training # wrappers over shell commands

from recsys_data import RecSysData

from gat import create_val_edge_batched, NodeIdMap, RecLightGCN, JustLightGCN, JustGAT, RecGAT, DotproductEdgePredictor, LinearEdgePredictor, BprLossLoader, test_out, l2_reg, train_test_split_pos_edges, int_tensor, unique_edges, loss_f, BprTraining, device

import pandas as pd
import numpy as np

def load_data(data_path, data_product_properties_path):
    """
    Load the input data from the specified path.
    """
    add_to_cart = pd.read_parquet(f'{data_path}/input/add_to_cart.parquet')
    page_visit = pd.read_parquet(f'{data_path}/input/page_visit.parquet')
    product_buy = pd.read_parquet(f'{data_path}/input/product_buy.parquet')
    product_properties = pd.read_parquet(f'{data_product_properties_path}/product_properties.parquet')
    remove_from_cart = pd.read_parquet(f'{data_path}/input/remove_from_cart.parquet')
    search_query = pd.read_parquet(f'{data_path}/input/search_query.parquet')

    return RecSysData(
        add_to_cart,
        page_visit,
        product_buy,
        product_properties,
        remove_from_cart,
        search_query
    )

def load_target(target_path):
  target_active_clients = np.load(f'{target_path}/active_clients.npy')
  target_popularity_propensity_category = np.load(f'{target_path}/popularity_propensity_category.npy')
  target_popularity_propensity_sku = np.load(f'{target_path}/popularity_propensity_sku.npy')
  target_propensity_category = np.load(f'{target_path}/propensity_category.npy')
  target_propensity_sku = np.load(f'{target_path}/propensity_sku.npy')

  target_train_target = pd.read_parquet(f'{target_path}/train_target.parquet')
  target_validation_target = pd.read_parquet(f'{target_path}/validation_target.parquet')

  return {
      'active_clients': target_active_clients,
      'popularity_propensity_category': target_popularity_propensity_category,
      'popularity_propensity_sku': target_popularity_propensity_sku,
      'propensity_category': target_propensity_category,
      'propensity_sku': target_propensity_sku,
      'train_target': target_train_target,
      'validation_target': target_validation_target
  }

def drop_duplicates(df, subset=['client_id', 'sku']):
  df = df.drop_duplicates(subset=subset)

def all_categories_in_df_numpy(df):
  all_categories = df['category'].unique()
  all_categories.sort()
  all_categories
  return all_categories

def all_users_in_df_numpy(df):
  return df['client_id'].unique()

def all_items_in_df_numpy(df):
  return df['sku'].unique()

def join_item_with_category(df, product_properties):
  """
  Joins the item dataframe with the category dataframe.
  """
  return df.merge(
    product_properties[['sku', 'category']], 
    on='sku', 
    how='left'
  )


# input_data.product_buy = input_data.product_buy[pd.to_datetime(input_data.product_buy['timestamp']) < first_train_timestamp]
# input_data.add_to_cart = input_data.add_to_cart[pd.to_datetime(input_data.add_to_cart['timestamp']) < first_train_timestamp]
# input_data.remove_from_cart = input_data.remove_from_cart[pd.to_datetime(input_data.remove_from_cart['timestamp']) < first_train_timestamp]