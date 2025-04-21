import pandas as pd
import numpy as np
import csv
import requests
from opensearchpy import OpenSearch, RequestsHttpConnection
from sentence_transformers import SentenceTransformer
import warnings
import urllib3
import time
import psutil
import os
import json

# 証明書の検証に関する警告を無視
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
warnings.filterwarnings("ignore")

# パフォーマンス計測用の関数
def measure_performance(func):
    def wrapper(*args, **kwargs):
        # 開始時間とCPU使用率を記録
        start_time = time.time()
        start_cpu = psutil.cpu_percent(interval=None)
        
        # 関数を実行
        result = func(*args, **kwargs)
        
        # 終了時間とCPU使用率を記録
        end_time = time.time()
        end_cpu = psutil.cpu_percent(interval=0.1)
        
        # 結果を表示
        print(f"\n===== パフォーマンス計測 ({func.__name__}) =====")
        print(f"実行時間: {end_time - start_time:.2f}秒")
        print(f"CPU使用率: {end_cpu:.1f}%")
        print("========================================\n")
        
        return result
    return wrapper

# OpenSearchの接続設定
host = 'localhost'
port = 9200
# OpenSearch 1.0.1のDockerイメージではセキュリティプラグインが無効になっている場合がある
# その場合は認証情報は不要
try_auth = True
auth = ('admin', 'admin')  # デフォルトのcredentials

# OpenSearchクライアントの設定
def create_client():
    try:
        # まず認証ありで接続を試みる
        if try_auth:
            client = OpenSearch(
                hosts=[{'host': host, 'port': port}],
                http_auth=auth,
                use_ssl=True,
                verify_certs=False,
                ssl_show_warn=False,
                connection_class=RequestsHttpConnection
            )
            # 接続テスト
            client.info()
            print("認証付きでOpenSearchに接続しました")
            return client
    except Exception as e:
        print(f"認証付きでの接続に失敗しました: {e}")
        print("認証なしでの接続を試みます...")
    
    # 認証なしで接続を試みる
    try:
        client = OpenSearch(
            hosts=[{'host': host, 'port': port}],
            use_ssl=False,  # HTTPでの接続を試みる
            connection_class=RequestsHttpConnection
        )
        # 接続テスト
        client.info()
        print("認証なしでOpenSearchに接続しました")
        return client
    except Exception as e:
        print(f"認証なしでの接続にも失敗しました: {e}")
        print("最後の手段として基本的な設定で接続を試みます...")
    
    # 最もシンプルな接続方法を試す
    client = OpenSearch(
        hosts=[f"{host}:{port}"]
    )
    return client

# クライアントの作成
client = create_client()

# Embeddingモデルのロード
print("Embeddingモデルをロード中...")
start_time = time.time()
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
print(f"モデルロード完了（所要時間: {time.time() - start_time:.2f}秒）")

# インデックス名 (先頭は小文字、特殊文字なしにしてOpenSearchの命名規則に合わせる)
index_name = 'qa-hybrid'  # ダッシュは問題ないが、アンダースコアよりも一般的

# OpenSearchのバージョンを確認
try:
    info = client.info()
    version = info.get('version', {}).get('number', '')
    print(f"OpenSearchのバージョン: {version}")
except Exception as e:
    print(f"OpenSearchのバージョン取得中にエラーが発生しました: {e}")
    print(f"エラー詳細: {str(e)}")
    version = '1.0.1'  # デフォルトでdocker-compose.ymlのバージョンを使用

@measure_performance
def semantic_search(query, top_k=3):
    # クエリのembedding生成
    embed_start = time.time()
    query_embedding = model.encode(query).tolist()
    print(f"クエリのエンベディング生成時間: {time.time() - embed_start:.3f}秒")
    
    try:
        # OpenSearch 1.0.1では高度なベクトル検索が使えないため、
        # テキスト検索に切り替える
        search_body = {
            "size": top_k,
            "query": {
                "hybrid": {
                  "queries": [
                    {
                      "match": {
                        "keyword": {
                          "query": query,
                          "analyzer": "kuromoji"
                        }
                      }
                    },
                    {
                      "knn": {
                        "vector": {
                          "vector": query_embedding,
                          "k": 5
                          }
                        }
                    }
                ]
            }
        }
        }
        
        search_start = time.time()
        response = client.search(
            body=search_body,
            index=index_name
        )
        print(f"テキスト検索実行時間: {time.time() - search_start:.3f}秒")
        
        if len(response['hits']['hits']) > 0:
            print(f"\n検索クエリ: '{query}' の結果:")
            for hit in response['hits']['hits']:
                print(f"スコア: {hit['_score']:.4f}, 質問: {hit['_source']['question']}")
                print(f"回答: {hit['_source']['answer']}\n")
        else:
            print(f"\n検索クエリ: '{query}' に一致する結果はありませんでした。")
    except Exception as e:
        print(f"検索中にエラーが発生しました: {e}")
        print(f"エラー詳細: {str(e)}")

# OpenSearchの状態を確認する関数
def check_opensearch_status():
    try:
        health = client.cluster.health()
        print("OpenSearchクラスタの状態:")
        print(f"  ステータス: {health.get('status', '不明')}")
        print(f"  ノード数: {health.get('number_of_nodes', '不明')}")
        
        # OpenSearch 1.1.0では一部のキーが異なる可能性があるため、安全にアクセス
        if 'number_of_indices' in health:
            print(f"  インデックス数: {health['number_of_indices']}")
        
        # シャード情報も安全にアクセス
        active_shards = health.get('active_shards', '不明')
        total_shards = health.get('total_shards', '不明')
        print(f"  シャード: {active_shards} (アクティブ)/{total_shards} (合計)")
        
        try:
            # ノード情報
            nodes_info = client.nodes.info()
            for node_id, node in nodes_info['nodes'].items():
                print(f"  ノード: {node.get('name', 'unknown')} (ID: {node_id})")
                print(f"    バージョン: {node.get('version', 'unknown')}")
                os_info = node.get('os', {})
                print(f"    OS: {os_info.get('name', 'unknown')} {os_info.get('version', 'unknown')}")
        except Exception as e:
            print(f"  ノード詳細情報の取得に失敗: {e}")
        
        try:
            # 利用可能なインデックスの一覧
            indices = client.indices.get("*")
            print("\n利用可能なインデックス:")
            for idx_name in indices:
                print(f"  - {idx_name}")
        except Exception as e:
            print(f"  インデックス一覧の取得に失敗: {e}")
        
        return True
    except Exception as e:
        print(f"OpenSearchの状態確認中にエラーが発生しました: {e}")
        print(f"エラー詳細: {str(e)}")
        # エラーがあっても継続する
        return True

# メイン処理
def main():
    # 全体の実行開始時間
    global_start_time = time.time()
    
    # 現在のプロセスのCPU使用率を監視開始
    process = psutil.Process(os.getpid())
    initial_cpu_usage = process.cpu_percent(interval=0.1)
    
    # OpenSearchの状態を確認
    print("\n=== OpenSearchの状態確認 ===")
    check_opensearch_status()
    print("OpenSearchの接続テストを試みます...")
    try:
        # シンプルな生存確認
        ping_result = client.ping()
        print(f"OpenSearchへの接続: {'成功' if ping_result else '失敗'}")
    except Exception as e:
        print(f"OpenSearchへの接続中にエラーが発生しましたが、処理を継続します: {e}")
    # テスト検索の実行
    test_query = "OpenSearchの特徴について教えて"
    semantic_search(test_query)
    
    # 全体の実行終了時間
    global_end_time = time.time()
    final_cpu_usage = process.cpu_percent(interval=0.1)
    
    # 総合結果
    print("\n========== 総合実行結果 ==========")
    print(f"総実行時間: {global_end_time - global_start_time:.2f}秒")
    print(f"最終CPU使用率: {final_cpu_usage:.1f}%")
    print(f"最大メモリ使用量: {process.memory_info().rss / (1024 * 1024):.1f} MB")
    print("====================================")

if __name__ == "__main__":
    main() 