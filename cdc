[1mdiff --git a/preprocess/generate_hypergraph.py b/preprocess/generate_hypergraph.py[m
[1mindex 17d8608..8069d3c 100644[m
[1m--- a/preprocess/generate_hypergraph.py[m
[1m+++ b/preprocess/generate_hypergraph.py[m
[36m@@ -1249,6 +1249,7 @@[m [mdef calculate_event_edge_weight([m
     - alpha_*: 各项特征的加权系数[m
 [m
     返回：边权（float），若无交集返回 None[m
[32m+[m[41m    [m
     """[m
     ents_i = {row_i['Source_name_encoded'], row_i['Target_name_encoded']}[m
     ents_j = {row_j['Source_name_encoded'], row_j['Target_name_encoded']}[m
