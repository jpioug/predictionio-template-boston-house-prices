# -*- coding: utf-8 -*-

"""
Import sample data for classification engine
"""

import predictionio
import argparse
import os
import sys
import pandas as pd

def import_events(client, path):
  df = pd.read_csv(path, skiprows=1)
  count = 0
  print("Importing data...")
  for data in df.to_dict('records'):
    client.create_event(event="$set",
                        entity_type="user",
                        entity_id=str(count), # use the count num as user ID
                        properties=data)
    count += 1
  print("%s events are imported." % count)

def download_datafile(url, path):
  if sys.version_info[0] >= 3:
    from urllib.request import urlretrieve
  else:
    from urllib import urlretrieve
  urlretrieve(url, path)

if __name__ == '__main__':
  parser = argparse.ArgumentParser(
    description="Import sample data")
  parser.add_argument('--access-key', dest='access_key', default='BHP_TOKEN')
  parser.add_argument('--url', default="http://localhost:7070")
  parser.add_argument('--file', dest='file_path', default="./data/boston_house_prices.csv")
  parser.add_argument('--download-url', dest='download_url',
    default="https://raw.githubusercontent.com/scikit-learn/scikit-learn/0.19.0/sklearn/datasets/data/boston_house_prices.csv")

  args = parser.parse_args()
  print(args)

  if not os.path.exists(args.file_path):
    download_datafile(args.download_url, args.file_path)

  client = predictionio.EventClient(
    access_key=args.access_key,
    url=args.url,
    threads=5,
    qsize=500)
  import_events(client, args.file_path)

