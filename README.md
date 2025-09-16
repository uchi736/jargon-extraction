# Jargon - 専門用語抽出システム

## 概要
Jargonは、PDF、Word、Markdownなどの文書から専門用語を自動的に抽出し、用語辞書を構築するPythonベースのシステムです。自然言語処理技術とLLM（大規模言語モデル）を活用して、高精度な専門用語の抽出と定義生成を行います。

## 主な機能
- 📄 複数形式の文書処理（PDF、DOCX、Markdown、HTML、TXT）
- 🔍 Azure OpenAI/OpenAI GPTを使用した専門用語抽出と定義生成
- 📊 TF-IDF、Perplexityベースの統計的用語抽出
- 🎯 SudachiPyによる高精度な日本語形態素解析
- 📈 リッチなコンソール出力とロギング
- ⚡ LLMとルールベースのハイブリッド抽出
- 📚 評価指標による用語の重要度スコアリング

## 必要要件
- Python 3.8以上
- Azure OpenAI APIキー または OpenAI APIキー（LLM機能を使用する場合）

## インストール

### 1. リポジトリのクローン
```bash
git clone https://github.com/yourusername/Jargon.git
cd Jargon
```

### 2. 仮想環境の作成と有効化
```bash
python -m venv myenv
# Windows
myenv\Scripts\activate
# macOS/Linux
source myenv/bin/activate
```

### 3. 依存パッケージのインストール
```bash
pip install -r requirements.txt
```

### 4. 環境変数の設定
`.env`ファイルを作成し、以下の内容を設定：
```env
# OpenAI API（オプション）
OPENAI_API_KEY=your_openai_api_key_here

# Google Gemini API（オプション - 使用する場合）
GOOGLE_API_KEY=your_google_api_key_here

# Azure OpenAI（オプション - Embeddingで使用）
AZURE_OPENAI_API_KEY=your_azure_key_here
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com
AZURE_OPENAI_API_VERSION=2024-12-01-preview
AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME=text-embedding-3-small

# Database（オプション）
DATABASE_URL=postgresql://user:password@localhost/jargon_db

# LangSmith（オプション - トレース用）
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_langsmith_key_here
LANGCHAIN_PROJECT=term-extraction
```

## 使用方法

### サンプルコードの実行
```bash
# 統計的手法による抽出（TF-IDF + 形態素解析）
python examples/statistical_extractor.py ./input ./output/statistical_terms.json

# LLMのみによる抽出
python examples/llm_extractor.py ./input ./output/llm_terms.json

# 改良版LLM抽出（より高度な検証機能付き）
python examples/llm_extractor_v2.py ./input ./output/llm_terms_v2.json
```

### 評価ツールの実行
```bash
# Azure OpenAIによるperplexity計算
python src/evaluation/azure_perplexity.py

# 入力ログ確率の計算
python src/evaluation/input_logprobs_calculator.py
```

## プロジェクト構造
```
Jargon/
├── src/                       # ソースコード
│   ├── core/                  # コア機能
│   │   ├── main.py           # メインアプリケーション
│   │   └── main_extractor.py # 主要抽出エンジン
│   ├── evaluation/            # 評価・スコアリング
│   │   ├── azure_perplexity.py        # Azure OpenAI perplexity計算
│   │   ├── enhanced_perplexity.py     # 拡張perplexity計算
│   │   ├── input_logprobs_calculator.py # 入力ログ確率計算
│   │   ├── mask_generator.py          # マスク生成
│   │   └── unknown_term_evaluator.py  # 未知語評価
│   ├── extraction/            # 抽出アルゴリズム
│   │   └── generic_perplexity_extractor.py # 汎用perplexity抽出
│   └── utils/                 # ユーティリティ
│       ├── document_loader.py  # 共通文書ローダー
│       ├── base_extractor.py   # 抽出器基底クラス
│       ├── corpus_idf.py       # IDF計算
│       └── preprocessor.py     # 前処理
├── examples/                   # サンプル実装
│   ├── statistical_extractor.py  # 統計的手法による抽出
│   ├── llm_extractor.py         # LLMのみによる抽出
│   └── llm_extractor_v2.py      # 改良版LLM抽出
├── input/                      # 入力文書ディレクトリ
├── output/                     # 出力ディレクトリ
├── logs/                       # ログファイル
├── docs/                       # ドキュメント
├── config.yml                  # 設定ファイル
├── requirements.txt            # Python依存パッケージ
├── .env                        # 環境変数
└── README.md                   # このファイル
```

## 設定
`config.yml`で以下の項目を設定可能：
- 抽出する用語の最小/最大文字数
- LLMのモデル選択
- ログレベル
- 出力フォーマット

## 技術的詳細

### 技術スタック

#### コア技術
- **Python 3.8+**: 非同期処理対応、型ヒント活用
- **asyncio/aiofiles**: 非同期I/O処理による高速化
- **Type Hints**: 静的型チェックによるコード品質向上

#### 文書処理
- **PyMuPDF**: PDF文書の高速解析とテキスト抽出
  - メタデータ抽出
  - レイアウト保持オプション
- **python-docx**: Word文書の構造化解析
  - スタイル情報の保持
  - テーブル・画像の処理
- **BeautifulSoup4**: HTMLの構造解析
- **Markdown**: Markdownパーサー

#### 自然言語処理（NLP）

##### 形態素解析
- **SudachiPy**: 日本語形態素解析器
  - A/B/Cモードによる解析粒度の調整
  - 専門用語・複合語の適切な分割

##### 専門用語抽出アルゴリズム
- **統計的手法**: TF-IDF、出現頻度ベースのフィルタリング
- **言語学的手法**: SudachiPyによる形態素解析、品詞パターンマッチング
- **LLMベース**: Azure OpenAI/GPTによる用語抽出と検証
- **Perplexityベース**: ログ確率を用いた専門用語スコアリング

#### LLM統合
- **Azure OpenAI API**
  - GPT-4o/GPT-4: 高精度な定義生成と用語抽出
  - text-embedding-3-small: 高性能な埋め込み生成
  - Function Calling: 構造化出力
- **OpenAI API（オプション）**
  - GPT-4: 高精度な定義生成
  - GPT-3.5-turbo: コスト効率的な処理
- **LangChain**: 
  - プロンプトテンプレート管理
  - チェーン構築（LCEL）
  - メモリ管理
  - ドキュメントローダー

#### ベクトル処理
- **text-embedding-3-small**: Azure OpenAIの最新埋め込みモデル（1536次元）
- **用語類似度計算**: コサイン類似度による関連用語の抽出

#### データ処理
- **scikit-learn**: TF-IDF計算、統計的特徴抽出
- **NumPy**: 数値計算とベクトル演算
- **Pandas**: データフレーム操作と分析

### パフォーマンス最適化

- **バッチ処理**: 大規模文書の分割処理
- **非同期処理**: asyncio/aiohttpによるI/O効率化
- **キャッシング**: 結果のメモリキャッシュ

### ログとデバッグ

- **Rich Console**: 視覚的に見やすいコンソール出力
- **ログファイル**: logs/ディレクトリに処理ログを保存
- **デバッグモード**: 詳細な実行情報の表示


## トラブルシューティング
- **MeCabエラー**: このプロジェクトはMeCab不要の軽量版です
- **メモリ不足**: 大きな文書の場合は、バッチサイズを調整してください
- **API制限**: OpenAI APIのレート制限に注意してください
- **GPU関連**: CUDAバージョンの不一致はPyTorchの再インストールで解決
- **文字エンコーディング**: UTF-8以外の文書はchardetで自動検出

## ライセンス
MIT License

## 貢献
プルリクエストを歓迎します。大きな変更の場合は、まずissueを開いて変更内容を議論してください。

## 作者
[Your Name]

## 謝辞
- OpenAI GPT モデル
- Hugging Face Transformers
- LangChain コミュニティ