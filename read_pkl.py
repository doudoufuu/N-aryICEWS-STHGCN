import pickle
import pandas as pd

def print_pkl_content(file_path, max_rows=5):
    """
    读取.pkl文件并打印字段名和前几行内容
    
    参数:
        file_path: .pkl文件路径
        max_rows: 要打印的最大行数(默认为5)
    """
    try:
        # 读取.pkl文件
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        
        print(f"\n{'='*50}")
        print(f"文件内容类型: {type(data)}")
        print(f"{'='*50}\n")
        
        # 情况1: 数据是Pandas DataFrame
        if isinstance(data, pd.DataFrame):
            print("数据结构: Pandas DataFrame")
            print(f"字段名(列名): {data.columns.tolist()}")
            print(f"\n前{max_rows}行内容:")
            print(data.head(max_rows))
        
        # 情况2: 数据是Pandas Series
        elif isinstance(data, pd.Series):
            print("数据结构: Pandas Series")
            print(f"Series名称: {data.name}")
            print(f"\n前{max_rows}个值:")
            print(data.head(max_rows))
        
        # 情况3: 数据是字典
        elif isinstance(data, dict):
            print("数据结构: 字典")
            print(f"字典键(字段名): {list(data.keys())}")
            
            # 打印字典中每个字段的前几行(如果是可迭代对象)
            print(f"\n各字段前{max_rows}个元素:")
            for key, value in data.items():
                print(f"\n字段: {key} (类型: {type(value)})")
                
                # 如果是Pandas对象
                if isinstance(value, (pd.DataFrame, pd.Series)):
                    print(value.head(max_rows))
                # 如果是numpy数组
                elif hasattr(value, 'shape'):  # 适用于numpy数组
                    print(f"形状: {value.shape}")
                    print(value[:max_rows] if len(value.shape) == 1 
                          else value[:max_rows, :])
                # 其他可迭代对象
                elif hasattr(value, '__iter__') and not isinstance(value, str):
                    print(list(value)[:max_rows])
                else:
                    print(value)
        
        # 情况4: 数据是列表或元组
        elif isinstance(data, (list, tuple)):
            print(f"数据结构: {type(data).__name__}")
            print(f"长度: {len(data)}")
            print(f"\n前{max_rows}个元素:")
            
            for i, item in enumerate(data[:max_rows]):
                print(f"\n元素{i} (类型: {type(item)})")
                
                # 如果是Pandas对象
                if isinstance(item, (pd.DataFrame, pd.Series)):
                    if isinstance(item, pd.DataFrame):
                        print(f"DataFrame列名: {item.columns.tolist()}")
                        print(item.head(2))  # 只打印前2行
                    else:
                        print(item.head(max_rows))
                # 其他情况
                else:
                    print(item)
        
        # 其他数据类型
        else:
            print("数据结构: 其他类型")
            print("\n内容预览:")
            print(str(data)[:500] + "...")  # 限制输出长度
    
    except FileNotFoundError:
        print(f"错误: 文件 {file_path} 不存在")
    except Exception as e:
        print(f"读取文件时出错: {str(e)}")

# 使用示例
if __name__ == "__main__":
    file_path = "/home/beihang/hsy/Spatio-Temporal-Hypergraph-Model/data/csv_events/preprocessed_1/label_encoding.pkl"
    print_pkl_content(file_path)