# OpenSearch QAデータアップロードツール

このスクリプトは、CSVファイルに保存されたQAデータを読み込み、エンベディングを生成してOpenSearchにインデックスします。

## 前提条件

- Python 3.7以上
- Docker環境（docker-composeでOpenSearchが実行されていること）
- 必要なPythonライブラリ（requirements.txtに記載）

## セットアップ

1. 必要なPythonライブラリをインストールします

```bash
pip install -r requirements.txt
```

2. OpenSearchを起動します（docker-compose.yml を使用）

```bash
docker-compose up -d
```

## 使用方法

1. QAデータがCSV形式で用意されていることを確認します（qa_data.csv）
2. スクリプトを実行します

```bash
python upload_qa_to_opensearch.py
```

3. データが正常にアップロードされると、スクリプトは自動的に簡単な検索テストを実行します

## スクリプトの内容

- CSVファイルからQAデータを読み込みます
- 質問文のエンベディングを生成します（sentence-transformersを使用）
- エンベディングとともにデータをOpenSearchにインデックスします
- サンプルとして、意味的検索（セマンティック検索）を実行します

## 注意事項

- OpenSearchへの接続設定は、デフォルトでlocalhost:9200、ユーザー名/パスワード: admin/adminを使用しています
- 必要に応じてスクリプト内の接続設定を変更してください
- HTTPSで接続するため、スクリプトでは証明書検証を無効にしています（開発環境向け） # opensearch-hybrid-search
