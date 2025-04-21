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
index_name = 'qa-hybrid'  # ハイブリッド検索用に新しいインデックス名を使用

# OpenSearchのバージョンを確認
try:
    info = client.info()
    version = info.get('version', {}).get('number', '')
    print(f"OpenSearchのバージョン: {version}")
except Exception as e:
    print(f"OpenSearchのバージョン取得中にエラーが発生しました: {e}")
    print(f"エラー詳細: {str(e)}")
    version = '1.0.1'  # デフォルトでdocker-compose.ymlのバージョンを使用

# インデックスの存在を確認する関数
def check_index_exists():
    try:
        return client.indices.exists(index=index_name)
    except Exception as e:
        print(f"インデックス確認中にエラーが発生しました: {e}")
        return False

# インデックスの詳細情報を取得
def get_index_info():
    try:
        if check_index_exists():
            try:
                # インデックスの統計情報を取得
                stats = client.indices.stats(index=index_name)
                docs_count = stats.get('_all', {}).get('primaries', {}).get('docs', {}).get('count', 0)
                size_in_bytes = stats.get('_all', {}).get('primaries', {}).get('store', {}).get('size_in_bytes', 0)
                size_in_mb = size_in_bytes / (1024 * 1024)
                
                print(f"インデックス '{index_name}' の情報:")
                print(f"  ドキュメント数: {docs_count}")
                print(f"  サイズ: {size_in_mb:.2f} MB")
            except Exception as e:
                print(f"インデックス統計情報の取得に失敗しました: {e}")
            
            try:
                # マッピング情報の取得
                mapping = client.indices.get_mapping(index=index_name)
                print(f"  マッピング: {json.dumps(mapping.get(index_name, {}).get('mappings', {}), indent=2)}")
            except Exception as e:
                print(f"マッピング情報の取得に失敗しました: {e}")
            
            return True
        else:
            print(f"インデックス '{index_name}' は存在しません")
            return False
    except Exception as e:
        print(f"インデックス情報取得中にエラーが発生しました: {e}")
        return False

# インデックスをリセットする関数
def reset_index():
    try:
        if check_index_exists():
            print(f"インデックス '{index_name}' を削除します...")
            client.indices.delete(index=index_name)
            print(f"インデックス '{index_name}' を削除しました")
            
            # 削除確認
            time.sleep(1)  # 少し待機してインデックス削除が反映されるのを待つ
            if not check_index_exists():
                print("インデックスは正常に削除されました")
            else:
                print("インデックスが削除されていません")
        else:
            print(f"インデックス '{index_name}' はまだ存在しないため、削除は不要です")
        return True
    except Exception as e:
        print(f"インデックス削除中にエラーが発生しました: {e}")
        print(f"エラー詳細: {str(e)}")
        return False

# kuromojiプラグインの存在を確認
def check_kuromoji_plugin():
    try:
        plugins = client.cat.plugins(format="json")
        has_kuromoji = any("analysis-kuromoji" in plugin.get("component", "") for plugin in plugins)
        if has_kuromoji:
            print("kuromoji プラグインが利用可能です")
            return True
        else:
            print("kuromoji プラグインが見つかりません")
            return False
    except Exception as e:
        print(f"プラグイン確認中にエラーが発生しました: {e}")
        return False

@measure_performance
def create_index():
    # インデックスが存在しない場合は作成
    if not check_index_exists():
        # エンベディングの次元数
        dims = 384  # 使用するモデルによって次元数が異なる場合は調整
        
        # kuromoji アナライザーが利用可能かチェック
        has_kuromoji = check_kuromoji_plugin()
        
        # 分析設定
        analysis_settings = {
            'analyzer': {
                'kuromoji_analyzer': {
                    'type': 'custom',
                    'tokenizer': 'kuromoji_tokenizer' if has_kuromoji else 'standard',
                    'filter': ['lowercase']
                },
                'japanese': {
                    'type': 'standard',
                    'stopwords': '_japanese_'
                }
            }
        }
        
        # ハイブリッド検索用インデックス設定
        # k-NNとテキスト検索の両方をサポート
        index_body = {
            'settings': {
                'index': {
                    'number_of_shards': 1,
                    'number_of_replicas': 0,  # 開発環境では複製なし
                    'knn': True,  # k-NNを有効化
                    'knn.space_type': 'cosinesimil'  # コサイン類似度を使用
                },
                'analysis': analysis_settings
            },
            'mappings': {
                'properties': {
                    'id': {'type': 'keyword'},
                    'question': {
                        'type': 'text',
                        'analyzer': 'kuromoji_analyzer' if has_kuromoji else 'japanese',
                        'fields': {
                            'keyword': {  # キーワード検索用のサブフィールド
                                'type': 'keyword',
                                'ignore_above': 256
                            }
                        }
                    },
                    'answer': {
                        'type': 'text',
                        'analyzer': 'kuromoji_analyzer' if has_kuromoji else 'japanese'
                    },
                    'vector': {  # ベクトル検索用のフィールド
                        'type': 'knn_vector',
                        'dimension': dims
                    }
                }
            }
        }
        
        # インデックス作成が失敗した場合の代替バージョン（より基本的な設定）
        fallback_index_body = {
            'settings': {
                'index': {
                    'number_of_shards': 1,
                    'number_of_replicas': 0
                },
                'analysis': {
                    'analyzer': {
                        'japanese': {
                            'type': 'standard',
                            'stopwords': '_japanese_'
                        }
                    }
                }
            },
            'mappings': {
                'properties': {
                    'id': {'type': 'keyword'},
                    'question': {
                        'type': 'text',
                        'analyzer': 'japanese'
                    },
                    'answer': {
                        'type': 'text',
                        'analyzer': 'japanese'
                    },
                    'vector_text': {  # ベクトルをJSONテキストとして保存
                        'type': 'text',
                        'index': False
                    }
                }
            }
        }
        
        # まずは通常のインデックスを作成
        try:
            print("ハイブリッド検索用インデックスを作成します...")
            client.indices.create(index=index_name, body=index_body)
            print(f"インデックス '{index_name}' を作成しました")
            
            # インデックスが作成できたことを確認
            if check_index_exists():
                print("インデックスが正常に作成されました。マッピングを確認します。")
                get_index_info()
                return True, True  # (成功, ベクトル検索対応)
            else:
                print("インデックスの作成に失敗しました。")
                return False, False
        except Exception as e:
            print(f"高度なインデックス作成に失敗しました: {e}")
            print("代替方法でインデックスを作成します...")
            
            try:
                client.indices.create(index=index_name, body=fallback_index_body)
                print(f"インデックス '{index_name}' を代替方法で作成しました")
                get_index_info()
                return True, False  # (成功, テキスト形式のみ)
            except Exception as e2:
                print(f"代替インデックス作成にも失敗しました: {e2}")
                return False, False
    else:
        print(f"インデックス '{index_name}' は既に存在します")
        get_index_info()
        
        # マッピング情報を確認してベクトル検索対応かをチェック
        try:
            mapping = client.indices.get_mapping(index=index_name)
            props = mapping.get(index_name, {}).get('mappings', {}).get('properties', {})
            has_vector = 'vector' in props and props['vector'].get('type') == 'knn_vector'
            return True, has_vector
        except Exception:
            return True, False

@measure_performance
def load_csv():
    # CSVファイルの読み込み
    try:
        qa_data = pd.read_csv('qa_data.csv')
        print(f"{len(qa_data)}件のQAデータを読み込みました")
        # データの先頭5行を表示して確認
        print("データサンプル:")
        print(qa_data.head(5).to_string())
        return qa_data
    except Exception as e:
        print(f"CSVの読み込みエラー: {e}")
        exit(1)

@measure_performance
def generate_embeddings_and_index(qa_data, has_vector_field=True):
    # データをOpenSearchにインデックス
    successful_uploads = 0
    
    # 各エンベディング生成の時間を記録
    embedding_times = []
    
    for _, row in qa_data.iterrows():
        try:
            # 質問文のembedding生成（時間計測）
            embed_start = time.time()
            question_embedding = model.encode(row['question']).tolist()
            embed_time = time.time() - embed_start
            embedding_times.append(embed_time)
            
            # ドキュメントの作成
            document = {
                'id': str(row['id']),
                'question': row['question'],
                'answer': row['answer']
            }
            
            # ベクトル形式またはテキスト形式でエンベディングを保存
            if has_vector_field:
                # knn_vectorフィールドとして保存
                document['vector'] = question_embedding
            else:
                # テキストとして保存（代替手段）
                document['vector_text'] = json.dumps(question_embedding)
            
            # OpenSearchにドキュメントをインデックス
            index_start = time.time()
            response = client.index(
                index=index_name,
                body=document,
                id=str(row['id']),
                refresh=True  # 即時反映
            )
            index_time = time.time() - index_start
            
            successful_uploads += 1
            print(f"ID {row['id']} のデータをインデックスしました（エンベディング: {embed_time:.3f}秒, インデックス: {index_time:.3f}秒）")
        except Exception as e:
            print(f"ID {row['id']} のインデックス中にエラーが発生しました: {e}")
            print(f"エラー詳細: {str(e)}")
    
    # エンベディング生成の平均時間を計算
    if embedding_times:
        avg_embedding_time = sum(embedding_times) / len(embedding_times)
        print(f"\nエンベディング生成の平均時間: {avg_embedding_time:.3f}秒/件")
    
    print(f"アップロード完了: {successful_uploads}/{len(qa_data)}件のデータがインデックスされました")
    
    # インデックスの詳細を表示
    get_index_info()
    
    return successful_uploads

@measure_performance
def test_hybrid_search(query, top_k=3):
    """ハイブリッド検索をテスト"""
    try:
        # クエリの埋め込みベクトルを生成
        embed_start = time.time()
        query_embedding = model.encode(query).tolist()
        embed_time = time.time() - embed_start
        print(f"クエリのエンベディング生成時間: {embed_time:.3f}秒")
        
        # マッピング情報を取得してベクトル検索対応かチェック
        try:
            mapping = client.indices.get_mapping(index=index_name)
            props = mapping.get(index_name, {}).get('mappings', {}).get('properties', {})
            has_vector = 'vector' in props and props['vector'].get('type') == 'knn_vector'
        except Exception:
            has_vector = False
        
        search_start = time.time()
        
        if has_vector:
            # OpenSearch 2.xのハイブリッド検索クエリを試行
            try:
                search_body = {
                    "size": top_k,
                    "query": {
                        "hybrid": {
                            "queries": [
                                {
                                    "match": {
                                        "question": {
                                            "query": query,
                                            "boost": 0.3
                                        }
                                    }
                                },
                                {
                                    "knn": {
                                        "vector": {
                                            "vector": query_embedding,
                                            "k": top_k
                                        }
                                    }
                                }
                            ]
                        }
                    }
                }
                
                response = client.search(
                    body=search_body,
                    index=index_name
                )
                print(f"ハイブリッド検索実行時間: {time.time() - search_start:.3f}秒")
                return response
            except Exception as e:
                print(f"ハイブリッド検索エラー: {e}")
                print("フォールバック: 個別の検索を実行します...")
        
        # フォールバック: 通常のマルチマッチクエリとkNN検索を別々に実行
        # まずはテキスト検索
        text_search_body = {
            "size": top_k,
            "query": {
                "match": {
                    "question": query
                }
            }
        }
        
        text_response = client.search(
            body=text_search_body,
            index=index_name
        )
        print(f"テキスト検索実行時間: {time.time() - search_start:.3f}秒")
        
        # ベクトル検索を試行（可能であれば）
        if has_vector:
            try:
                vector_search_body = {
                    "size": top_k,
                    "query": {
                        "knn": {
                            "vector": {
                                "vector": query_embedding,
                                "k": top_k
                            }
                        }
                    }
                }
                
                vector_response = client.search(
                    body=vector_search_body,
                    index=index_name
                )
                print(f"ベクトル検索実行時間: {time.time() - (search_start + (time.time() - search_start)):.3f}秒")
                
                # 結果をマージ（単純に結合して重複を除去）
                merged_hits = []
                doc_ids = set()
                
                # テキスト検索結果を追加
                for hit in text_response['hits']['hits']:
                    doc_ids.add(hit['_id'])
                    merged_hits.append(hit)
                
                # ベクトル検索結果を追加（重複を除去）
                for hit in vector_response['hits']['hits']:
                    if hit['_id'] not in doc_ids:
                        merged_hits.append(hit)
                
                # マージした結果を返す
                response = {
                    'hits': {
                        'total': {'value': len(merged_hits)},
                        'max_score': max([hit['_score'] for hit in merged_hits]) if merged_hits else 0,
                        'hits': merged_hits[:top_k]  # 上位k件を返す
                    }
                }
                
                return response
            except Exception as e:
                print(f"ベクトル検索エラー: {e}")
                print("テキスト検索結果のみを返します")
                return text_response
        else:
            # ベクトル検索がサポートされていない場合はテキスト検索結果のみを返す
            return text_response
        
    except Exception as e:
        print(f"検索中にエラーが発生しました: {e}")
        print(f"エラー詳細: {str(e)}")
        return {"hits": {"hits": []}}

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
    
    # 既存のインデックスをリセット（オプション）
    reset_index()
    
    # インデックスの作成
    success, has_vector = create_index()
    if not success:
        print("インデックスの作成に失敗しました。処理を継続します。")
    
    print(f"ベクトル検索対応: {'可能' if has_vector else '不可'}")
    
    # CSVファイルの読み込み
    qa_data = load_csv()
    
    # エンベディング生成とインデックス
    generate_embeddings_and_index(qa_data, has_vector)
    
    # インデックスが正しく作成されたか確認
    print("\n=== インデックスの確認 ===")
    get_index_info()
    
    # テスト検索の実行
    print("\n=== ハイブリッド検索のテスト ===")
    test_query = "OpenSearchの特徴について教えて"
    
    response = test_hybrid_search(test_query)
    
    # 結果の表示
    if len(response['hits']['hits']) > 0:
        print(f"\n検索クエリ: '{test_query}' の結果:")
        for hit in response['hits']['hits']:
            print(f"スコア: {hit['_score']:.4f}, 質問: {hit['_source']['question']}")
            print(f"回答: {hit['_source']['answer']}\n")
    else:
        print(f"\n検索クエリ: '{test_query}' に一致する結果はありませんでした。")
    
    # 全体の実行終了時間
    global_end_time = time.time()
    final_cpu_usage = process.cpu_percent(interval=0.1)
    
    # 総合結果
    print("\n========== 総合実行結果 ==========")
    print(f"総実行時間: {global_end_time - global_start_time:.2f}秒")
    print(f"最終CPU使用率: {final_cpu_usage:.1f}%")
    print(f"最大メモリ使用量: {process.memory_info().rss / (1024 * 1024):.1f} MB")
    print("====================================")
    
    # OpenSearch Dashboardsへのアクセス方法を表示
    print("\nOpenSearch Dashboardsへのアクセス:")
    print("1. ブラウザで http://localhost:5601 にアクセスしてください")
    print("2. デフォルトのユーザー名/パスワード: admin/admin")
    print("3. [インデックスパターン] で 'qa-hybrid*' と入力して作成")
    print("4. [Discovery] メニューからデータを参照できます")
    
    # ハイブリッド検索のAPI呼び出し例を表示
    print("\nハイブリッド検索のAPI呼び出し例:")
    print("```")
    print(f"GET {index_name}/_search")
    print("{")
    print('  "query": {')
    print('    "hybrid": {')
    print('      "queries": [')
    print('        {')
    print('          "match": {')
    print('            "question": {')
    print('              "query": "検索クエリ",')
    print('              "boost": 0.3')
    print('            }')
    print('          }')
    print('        },')
    print('        {')
    print('          "knn": {')
    print('            "vector": {')
    print('              "vector": [...エンベディングベクトル...],')
    print('              "k": 5')
    print('            }')
    print('          }')
    print('        }')
    print('      ]')
    print('    }')
    print('  }')
    print('}')
    print("```")

if __name__ == "__main__":
    main() 