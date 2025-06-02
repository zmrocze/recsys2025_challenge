

import numpy as np
import os
import matplotlib.pyplot as plt

# Helper function to calculate interaction statistics
def calculate_interaction_stats(df, interaction_name):
  interactions_per_user = df.groupby('client_id').size()
  time_range = (df['timestamp'].min(), df['timestamp'].max())
  stats = {
    'name': interaction_name,
    'total_interactions': len(df),
    'total_users': len(interactions_per_user),
    'avg_interactions_per_user': interactions_per_user.mean(),
    'avg_interactions_squared_per_user': (interactions_per_user ** 2).mean(),
    'min_interactions_per_user': interactions_per_user.min(),
    'max_interactions_per_user': interactions_per_user.max(),
    'median_interactions_per_user': interactions_per_user.median(),
    'std_interactions_per_user': interactions_per_user.std(),
    'time_range': time_range
  }
  
  # Add product-related statistics if 'sku' column exists
  if 'sku' in df.columns:
    products_per_user = df.groupby('client_id')['sku'].nunique()
    stats.update({
      'unique_products': df['sku'].nunique(),
      'product_interactions': products_per_user.sum(),
      'avg_products_per_user': products_per_user.mean(),
      'avg_products_squared_per_user': (products_per_user ** 2).mean(),
      'min_products_per_user': products_per_user.min(),
      'max_products_per_user': products_per_user.max(),
      'median_products_per_user': products_per_user.median(),
      'std_products_per_user': products_per_user.std(),
    })
  
  return stats, interactions_per_user

# Helper function to plot distribution
def plot_interaction_distribution(interactions_per_user, title, subplot_pos=None):
  if subplot_pos:
    plt.subplot(subplot_pos)
  plt.hist(interactions_per_user, bins=50, alpha=0.7, edgecolor='black', log=True)
  plt.xlabel('Number of Interactions per User')
  plt.ylabel('Number of Users')
  plt.title(title)


class RecSysData:
  def __init__(self, add_to_cart, page_visit, product_buy, product_properties, remove_from_cart, search_query):
    self.add_to_cart = add_to_cart
    self.page_visit = page_visit
    self.product_buy = product_buy
    self.product_properties = product_properties
    self.remove_from_cart = remove_from_cart
    self.search_query = search_query
    self.set_datasets()
  
  def set_datasets(self):
    self.datasets = {
      'add_to_cart': self.add_to_cart,
      'page_visit': self.page_visit,
      'product_buy': self.product_buy,
      'product_properties': self.product_properties,
      'remove_from_cart': self.remove_from_cart,
      'search_query': self.search_query
    }
    self.user_indexed_datasets = {i:self.datasets[i] for i in self.datasets if i!='product_properties'}

  def __repr__(self):
    return (f"RecSysData(\n"
        f"  add_to_cart={len(self.add_to_cart)},\n"
        f"  page_visit={len(self.page_visit)},\n"
        f"  product_buy={len(self.product_buy)},\n"
        f"  product_properties={len(self.product_properties)},\n"
        f"  remove_from_cart={len(self.remove_from_cart)},\n"
        f"  search_query={len(self.search_query)},\n"

        f"  all users: {len(self.all_users_set())},\n"
        f"  all products: {len(self.all_products_set())}\n"
        f"  all urls: {len(self.all_urls_set())},\n"
        f")")

  def all_users_set(self):
    all_users = set(self.add_to_cart['client_id']).union(
        set(self.page_visit['client_id']),
        set(self.product_buy['client_id']),
        set(self.remove_from_cart['client_id']),
        set(self.search_query['client_id'])
    )
    return all_users
  
  def all_urls_set(self):
    all_urls = set(self.page_visit['url']).union(
        set(self.search_query['query'])
    )
    return all_urls

  def all_products_set(self):
    all_products = set(self.add_to_cart['sku']).union(
        set(self.product_buy['sku']),
        set(self.remove_from_cart['sku']),
    )
    return all_products

  def sampled_subset(self, sampled_users, reset_index=True):
    sampled_users_set = set(sampled_users)
    if reset_index:
      f = lambda x: x.reset_index(drop=True)
    else:
      f = lambda x: x
    a = RecSysData(
      f(self.add_to_cart[self.add_to_cart['client_id'].isin(sampled_users_set)]),
      f(self.page_visit[self.page_visit['client_id'].isin(sampled_users_set)]),
      f(self.product_buy[self.product_buy['client_id'].isin(sampled_users_set)]),
      self.product_properties,
      f(self.remove_from_cart[self.remove_from_cart['client_id'].isin(sampled_users_set)]),
      f(self.search_query[self.search_query['client_id'].isin(sampled_users_set)])
    )
    remaining_products = a.all_products_set()
    a.product_properties = a.product_properties[a.product_properties['sku'].isin(remaining_products)]
    a.set_datasets()
    return a

  def save_parquet(self, dir_path):
    if not os.path.exists(dir_path):
      os.makedirs(dir_path)
    self.add_to_cart.to_parquet(os.path.join(dir_path, 'add_to_cart.parquet'), index=False)
    self.page_visit.to_parquet(os.path.join(dir_path, 'page_visit.parquet'), index=False)
    self.product_buy.to_parquet(os.path.join(dir_path, 'product_buy.parquet'), index=False)
    self.remove_from_cart.to_parquet(os.path.join(dir_path, 'remove_from_cart.parquet'), index=False)
    self.search_query.to_parquet(os.path.join(dir_path, 'search_query.parquet'), index=False)
  
  def print_datasets_stats_nicely(self, title_name):
    # Calculate statistics for all datasets
    title_name = title_name.upper()
    print(f"=== {title_name} STATISTICS ===")
    # all_stats = []
    all_distributions = []

    for name, df  in self.user_indexed_datasets.items():
      stats, distribution = calculate_interaction_stats(df, name)
      # all_stats.append(stats)
      all_distributions.append(distribution)
      
      print(f"\n{name}:")
      print(f"  Total interactions: {stats['total_interactions']:,}")
      print(f"  Total users: {stats['total_users']:,}")
      print(f"  Average interactions per user: {stats['avg_interactions_per_user']:.2f}")
      print(f"  Average interactions squared per user: {stats['avg_interactions_squared_per_user']:.2f}")
      print(f"  Min interactions per user: {stats['min_interactions_per_user']}")
      print(f"  Max interactions per user: {stats['max_interactions_per_user']:,}")
      print(f"  Median interactions per user: {stats['median_interactions_per_user']:.2f}")
      print(f"  Std interactions per user: {stats['std_interactions_per_user']:.2f}")
      print(f"  Time range: {stats['time_range'][0]} to {stats['time_range'][1]}")
      if 'unique_products' in stats:
        print(f"  Unique products: {stats['unique_products']:,}")
        print(f"  Product interactions: {stats['product_interactions']:,}")
        print(f"  Average products per user: {stats['avg_products_per_user']:.2f}")
        print(f"  Average products squared per user: {stats['avg_products_squared_per_user']:.2f}")
        print(f"  Min products per user: {stats['min_products_per_user']}")
        print(f"  Max products per user: {stats['max_products_per_user']:,}")
        print(f"  Median products per user: {stats['median_products_per_user']:.2f}")
        print(f"  Std products per user: {stats['std_products_per_user']:.2f}")

    # Plot distributions for full dataset
    plt.figure(figsize=(15, 10))
    for i, (distribution, (name, _)) in enumerate(zip(all_distributions, self.datasets.items())):
      plot_interaction_distribution(distribution, f'{name} Distribution', 231 + i)

    plt.tight_layout()
    plt.suptitle(f'{title_name}: Distribution of Interactions per User', y=1.02, fontsize=16)
    plt.show()
