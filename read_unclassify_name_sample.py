#读/home/beihang/hsy/Spatio-Temporal-Hypergraph-Model/data/csv_events/preprocessed_1/unclassify_sourcename_data.csv文件，打印“ID”列的范围
import pandas as pd
data = pd.read_csv('/home/beihang/hsy/Spatio-Temporal-Hypergraph-Model/data/csv_events/preprocessed_1/unclassify_sourcename_data.csv')
print(f"[L0] data shape={data.shape}")
print(f"[L0] ID range: {data['ID'].min()} ~ {data['ID'].max()}")
