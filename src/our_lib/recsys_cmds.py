
import os
import sys

# to import validate_and_load_embeddings
# module_path = os.path.abspath(os.path.dirname(os.path.abspath(__file__)).join('../../.'))
# module_path = os.path.abspath(os.path.join('../../.')) # or the path to your source code
# sys.path.append(module_path)
# module_path = os.path.abspath(os.path.join('../recsys2025')) # or the path to your source code
# sys.path.append(module_path)

from recsys2025.validator.validate import validate_and_load_embeddings

import subprocess
from pathlib import Path

# recsys2025
# ! PYTHONPATH=$PYTHONPATH:../recsys2025 python -m data_utils.split_data --challenge-data-dir ../ubc_data_dirty
def split_data(challenge_data_dir):
  """
  Call the data_utils.split_data module with the specified challenge data directory.
  
  Args:
    challenge_data_dir (str): Path to the challenge data directory
  """
  here = Path.cwd()
  challenge_data_dir = str((here / challenge_data_dir).resolve())
  cmd = [
    'python', '-m', 'data_utils.split_data', 
    '--challenge-data-dir', challenge_data_dir
  ]

  return run_recsys_command(cmd)

# ! PYTHONPATH=$PYTHONPATH:../recsys2025 python -m baseline.aggregated_features_baseline.create_embeddings --data-dir ../ubc_data_dirty/ --embeddings-dir ../baseline_embeddings --num-days 1 7 30
def create_baseline_embeddings(data_dir, embeddings_dir, num_days=[1, 7, 30], top_n=None):
  """
  Call the baseline.aggregated_features_baseline.create_embeddings module with the specified parameters.
  
  --data-dir DATA_DIR   Directory with input and target data produced by data_utils.split_data
  --embeddings-dir EMBEDDINGS_DIR
                        Directory where to store generated embeddings
  --num-days [NUM_DAYS ...]
                        Numer of days to compute features
  --top-n TOP_N         Number of top column values to consider in feature generation
  """
  here = Path.cwd()
  data_dir = str((here / Path(data_dir)).resolve())
  embeddings_dir = str((here / Path(embeddings_dir)).resolve())
  cmd = [
    'python', '-m', 'baseline.aggregated_features_baseline.create_embeddings',
    '--data-dir', data_dir,
    '--embeddings-dir', embeddings_dir,
    '--num-days'
  ] + [str(day) for day in num_days] + (['--top-n', str(top_n)] if top_n else [])
  
  return run_recsys_command(cmd)

# python -m training_pipeline.train --data-dir <your_splitted_challenge_data_dir> --embeddings-dir <your-embeddings-dir> --tasks churn propensity_category propensity_sku --log-name <my_experiment> --accelerator gpu --devices 0 --neptune-api-token <your-api-token> --neptune-project <your-worskspace>/<your-project> --disable-relevant-clients-check
def contest_training(data_dir, embeddings_dir, score_dir, disable_relevant_clients_check=False, accelerator="gpu", devices="0", num_workers=None , tasks=None, log_name="my_experiment",  neptune_api_token=None, neptune_project=None):
  """
  Call the training_pipeline.train module with the specified parameters.
  
  Args:
    --data-dir DATA_DIR   Directory where target and input data are stored
    --embeddings-dir EMBEDDINGS_DIR
                          Directory where input embeddings are stored
    --tasks TASKS [TASKS ...]
                          Name of the task to train out of: churn propensity_category propensity_sku
    --log-name LOG_NAME   Experiment name
    --num-workers NUM_WORKERS
                          Number of subprocesses to use for data loading
    --accelerator ACCELERATOR
                          Accelerator type
    --devices [DEVICES ...]
                          List of devices to use. Possible options: "auto", id of single device to use or list of ids
                          of devices to use.
    --neptune-api-token NEPTUNE_API_TOKEN
                          Neptune API token.
    --neptune-project NEPTUNE_PROJECT
                          Name of Neptune project within workspace to save result to.
    --score-dir SCORE_DIR
                          Path to directory where to save best scores for each task
    --disable-relevant-clients-check
                          Disables relevant clients check in validator, but enables embeddings for sets of clients
                          other than relevant clients.
  """
  if tasks is None:
    tasks = ['churn', 'propensity_category', 'propensity_sku']
  
  here = Path.cwd()
  data_dir = str((here / Path(data_dir)).resolve())
  embeddings_dir = str((here / Path(embeddings_dir)).resolve())
  score_dir = str((here / Path(score_dir)).resolve())
  
  cmd = [
    'python', '-m', 'training_pipeline.train',
    '--data-dir', data_dir,
    '--embeddings-dir', embeddings_dir,
    '--tasks'
  ] + tasks + (['--num-workers', str(num_workers)] if num_workers else []) + [
    '--log-name', log_name,
    '--accelerator', accelerator,
    '--devices', devices
  ]
  
  if neptune_api_token:
    cmd.extend(['--neptune-api-token', neptune_api_token])
  
  if neptune_project:
    cmd.extend(['--neptune-project', neptune_project])
  
  if disable_relevant_clients_check:
    cmd.append('--disable-relevant-clients-check')
  
  return run_recsys_command(cmd)

recsys_path = (Path.cwd() / '../../recsys2025').resolve()

def run_recsys_command(cmd):
  env = os.environ.copy()
  env['PYTHONPATH'] = f"{env.get('PYTHONPATH', '')}:{str(recsys_path)}"

  print(f"Running command: {' '.join(cmd)}")
  print(f"recsys_path: {recsys_path}")

  result = subprocess.run(cmd, env=env, capture_output=True, text=True)
  
  if result.stdout:
    print(f"Stdout: {result.stdout}")
  if result.stderr:
    print(f"Stderr: {result.stderr}")
  if result.returncode != 0:
    print("Command failed!")
  
  return result