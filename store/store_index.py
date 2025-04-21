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
import logging

# ロガーの設定
logger = logging.getLogger("opensearch_index")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(levelname)s: %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

# 証明書の検証に関する警告を無視
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
warnings.filterwarnings("ignore")

# パフォーマンス計測用の関数
def measure_performance(func):
    def wrapper(*args, **kwargs):
        # 開始時間を記録
        start_time = time.time()
        
        # 関数を実行
        result = func(*args, **kwargs)
        
        # 終了時間を記録
        end_time = time.time()
        exec_time = end_time - start_time
        
        # DEBUGレベルで実行時間のみを記録
        logger.debug(f"{func.__name__}: 実行時間 {exec_time:.2f}秒")
        
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
            logger.info("認証付きでOpenSearchに接続しました")
            return client
    except Exception as e:
        logger.warning(f"認証付きでの接続に失敗: {str(e)[:100]}...")
    
    # 認証なしで接続を試みる
    try:
        client = OpenSearch(
            hosts=[{'host': host, 'port': port}],
            use_ssl=False,  # HTTPでの接続を試みる
            connection_class=RequestsHttpConnection
        )
        # 接続テスト
        client.info()
        logger.info("認証なしでOpenSearchに接続しました")
        return client
    except Exception as e:
        logger.warning(f"認証なしでの接続に失敗: {str(e)[:100]}...")
    
    # 最もシンプルな接続方法を試す
    logger.info("基本接続を試みます")
    client = OpenSearch(
        hosts=[f"{host}:{port}"]
    )
    return client

# クライアントの作成
client = create_client()

# Embeddingモデルのロード
logger.info("Embeddingモデルをロード中...")
start_time = time.time()
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
logger.info(f"モデルロード完了（{time.time() - start_time:.2f}秒）")

# インデックス名 (先頭は小文字、特殊文字なしにしてOpenSearchの命名規則に合わせる)
index_name = 'qa-hybrid'  # ハイブリッド検索用に新しいインデックス名を使用

# OpenSearchのバージョンを確認
try:
    info = client.info()
    version = info.get('version', {}).get('number', '')
    logger.info(f"OpenSearchのバージョン: {version}")
except Exception as e:
    logger.error(f"バージョン取得エラー: {str(e)[:100]}...")
    version = '1.0.1'  # デフォルトでdocker-compose.ymlのバージョンを使用

# インデックスの存在を確認する関数
def check_index_exists():
    try:
        return client.indices.exists(index=index_name)
    except Exception as e:
        logger.error(f"インデックス確認エラー: {str(e)[:100]}...")
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
                
                logger.info(f"インデックス '{index_name}': {docs_count}件, {size_in_mb:.2f}MB")
                
                # マッピング情報はDEBUGレベルでのみ表示
                mapping = client.indices.get_mapping(index=index_name)
                logger.debug(f"マッピング: {json.dumps(mapping.get(index_name, {}).get('mappings', {}), indent=2)}")
                
            except Exception as e:
                logger.warning(f"インデックス情報取得エラー: {str(e)[:100]}...")
            
            return True
        else:
            logger.info(f"インデックス '{index_name}' は存在しません")
            return False
    except Exception as e:
        logger.error(f"インデックス情報取得エラー: {str(e)[:100]}...")
        return False

# インデックスをリセットする関数
def reset_index():
    try:
        if check_index_exists():
            logger.info(f"インデックス '{index_name}' を削除します")
            client.indices.delete(index=index_name)
            
            # 削除確認
            time.sleep(1)
            if not check_index_exists():
                logger.info("インデックスは正常に削除されました")
            else:
                logger.warning("インデックスが削除されていません")
        else:
            logger.info(f"インデックス '{index_name}' は存在しないため、削除は不要です")
        return True
    except Exception as e:
        logger.error(f"インデックス削除エラー: {str(e)[:100]}...")
        return False

# kuromojiプラグインの存在を確認
def check_kuromoji_plugin():
    try:
        plugins = client.cat.plugins(format="json")
        has_kuromoji = any("analysis-kuromoji" in plugin.get("component", "") for plugin in plugins)
        if has_kuromoji:
            logger.info("kuromoji プラグインが利用可能です")
            return True
        else:
            logger.info("kuromoji プラグインが見つかりません")
            return False
    except Exception as e:
        logger.warning(f"プラグイン確認エラー: {str(e)[:100]}...")
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
            logger.info("ハイブリッド検索用インデックスを作成します")
            client.indices.create(index=index_name, body=index_body)
            
            if check_index_exists():
                logger.info("インデックスが正常に作成されました")
                get_index_info()
                return True, True  # (成功, ベクトル検索対応)
            else:
                logger.error("インデックスの作成に失敗しました")
                return False, False
        except Exception as e:
            logger.warning(f"高度なインデックス作成に失敗: {str(e)[:100]}...")
            logger.info("代替方法でインデックスを作成します")
            
            try:
                client.indices.create(index=index_name, body=fallback_index_body)
                logger.info("代替方法でインデックスを作成しました")
                get_index_info()
                return True, False  # (成功, テキスト形式のみ)
            except Exception as e2:
                logger.error(f"代替インデックス作成にも失敗: {str(e2)[:100]}...")
                return False, False
    else:
        logger.info(f"インデックス '{index_name}' は既に存在します")
        get_index_info()
        
        # マッピング情報を確認してベクトル検索対応かをチェック
        try:
            mapping = client.indices.get_mapping(index=index_name)
            props = mapping.get(index_name, {}).get('mappings', {}).get('properties', {})
            has_vector = 'vector' in props and props['vector'].get('type') == 'knn_vector'
            logger.info(f"ベクトル検索対応: {'可能' if has_vector else '不可'}")
            return True, has_vector
        except Exception:
            logger.warning("マッピング情報の取得に失敗しました")
            return True, False

@measure_performance
def load_csv():
    # CSVファイルの読み込み
    try:
        qa_data = pd.read_csv('qa_data.csv')
        logger.info(f"{len(qa_data)}件のQAデータを読み込みました")
        # サンプルはDEBUGレベルで表示
        logger.debug(f"データサンプル:\n{qa_data.head(3).to_string()}")
        return qa_data
    except Exception as e:
        logger.error(f"CSVの読み込みエラー: {str(e)[:100]}...")
        exit(1)

@measure_performance
def generate_embeddings_and_index(qa_data, has_vector_field=True):
    # データをOpenSearchにインデックス
    successful_uploads = 0
    total_records = len(qa_data)
    
    # 進行状況を表示するためのインターバル
    progress_interval = max(1, total_records // 10)
    
    logger.info(f"エンベディング生成とインデックス登録を開始（{total_records}件）")
    
    for i, (_, row) in enumerate(qa_data.iterrows()):
        try:
            # 質問文のembedding生成
            question_embedding = model.encode(row['question']).tolist()
            
            # ドキュメントの作成
            document = {
                'id': str(row['id']),
                'question': row['question'],
                'answer': row['answer']
            }
            
            # エンベディングを保存
            if has_vector_field:
                document['vector'] = question_embedding
            else:
                document['vector_text'] = json.dumps(question_embedding)
            
            # OpenSearchにドキュメントをインデックス
            client.index(
                index=index_name,
                body=document,
                id=str(row['id']),
                refresh=True
            )
            
            successful_uploads += 1
            
            # 進行状況の表示（10%ごと）
            if (i+1) % progress_interval == 0 or i+1 == total_records:
                progress = (i+1) / total_records * 100
                logger.info(f"進捗: {progress:.1f}%（{i+1}/{total_records}件）")
                
        except Exception as e:
            logger.error(f"ID {row['id']} のインデックスエラー: {str(e)[:100]}...")
    
    # 完了メッセージ
    logger.info(f"アップロード完了: {successful_uploads}/{total_records}件のデータをインデックスしました")
    
    # インデックス情報の表示
    get_index_info()
    
    return successful_uploads

# OpenSearchの状態を確認する関数
def check_opensearch_status():
    try:
        health = client.cluster.health()
        status = health.get('status', '不明')
        node_count = health.get('number_of_nodes', '不明')
        
        logger.info(f"OpenSearchクラスタ状態: {status}, ノード数: {node_count}")
        
        # インデックス一覧
        try:
            indices = client.indices.get("*")
            if indices:
                index_list = ", ".join(indices.keys())
                logger.info(f"利用可能なインデックス: {index_list}")
        except Exception as e:
            logger.warning(f"インデックス一覧取得エラー: {str(e)[:100]}...")
        
        return True
    except Exception as e:
        logger.error(f"クラスタ状態確認エラー: {str(e)[:100]}...")
        return False

# メイン処理
def main():
    # コマンドライン引数の解析
    import argparse
    parser = argparse.ArgumentParser(description='OpenSearch QAデータインデックス作成')
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        default='INFO', help='ログレベル設定')
    parser.add_argument('--reset', action='store_true', help='既存のインデックスを削除')
    args = parser.parse_args()
    
    # ログレベル設定
    if args.log_level:
        logger.setLevel(getattr(logging, args.log_level))
    
    logger.info("OpenSearch インデックス作成を開始します")
    
    # OpenSearchの状態確認
    check_opensearch_status()
    
    # インデックスのリセット（オプション）
    if args.reset:
        reset_index()
    
    # インデックスの作成
    success, has_vector = create_index()
    if not success:
        logger.warning("インデックスの作成に問題がありましたが、処理を継続します")
    
    # CSVファイルの読み込み
    qa_data = load_csv()
    
    # エンベディング生成とインデックス
    generate_embeddings_and_index(qa_data, has_vector)
    
    logger.info("インデックス作成処理が完了しました")

if __name__ == "__main__":
    main()