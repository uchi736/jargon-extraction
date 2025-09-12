# ファイル構成説明書

## 現在のディレクトリ構造

```
Jargon/
├── src/                        # メインソースコード
│   ├── core/                   # コアシステム
│   │   ├── main.py             # メインアプリケーション
│   │   └── main_extractor.py   # 統合型専門用語抽出システム
│   │
│   ├── evaluation/             # 評価手法モジュール
│   │   ├── azure_perplexity.py         # Azure OpenAI logprobs真のPerplexity計算
│   │   ├── enhanced_perplexity.py      # 強化版Perplexity（ホットスポット検出）
│   │   ├── input_logprobs_calculator.py # 入力ログ確率計算
│   │   └── mask_generator.py           # MASK文脈生成戦略
│   │
│   ├── extraction/             # 抽出手法モジュール
│   │   └── generic_perplexity_extractor.py # 汎用perplexity抽出器
│   │
│   └── utils/                  # ユーティリティ
│       ├── document_loader.py  # 共通文書ローダー
│       └── base_extractor.py   # 抽出器基底クラス
│
├── tests/                      # テスト関連
│   └── test_data/              # テストデータ
│       ├── final_test.json
│       ├── fixed_results.json
│       ├── improved_results.json
│       ├── legal_text_tokenization.json
│       └── test_output.json
│
├── examples/                   # サンプル実装
│   ├── statistical_extractor.py # 統計的手法による抽出（TF-IDF + 形態素解析）
│   └── llm_extractor.py        # LLMのみによる抽出（Gemini-2.0）
│
├── logs/                       # ログファイル
│   └── term_extraction.log    # 抽出処理ログ
│
├── docs/                       # ドキュメント
│   ├── azure_perplexity_detailed.md  # Azure Perplexity詳細仕様
│   ├── evaluation_logic_spec.md      # 評価ロジック仕様書
│   └── FILE_STRUCTURE.md             # 本ファイル
│
├── input/                      # 入力文書ディレクトリ
│   └── *.pdf                  # 処理対象のPDF文書
│
├── output/                     # 実行結果出力
│   └── generic_results.json   # 生成された抽出結果
│
├── old/                        # アーカイブ済みコード
│   └── rag_extractor.py       # 旧RAG統合型抽出（ChromaDB + LCEL）
│
├── myenv/                      # Python仮想環境
├── config.yml                  # 設定ファイル
├── requirements.txt            # 依存パッケージ
├── .env                        # 環境変数設定
├── .gitignore                  # Git除外設定
├── logprobs_calculation_logic.md  # ログ確率計算ロジックの説明
├── 計画書.md                   # プロジェクト計画書
└── README.md                   # プロジェクト説明
```

## 各ファイルの役割

### 🎯 メインシステム（src/core/）

**main.py**
- PyMuPDFを使用したPDF処理
- 各種文書フォーマット（PDF、Word、HTML、Markdown）の読み込み
- 非同期処理による効率化

**main_extractor.py**
- 本システムの中核
- 全機能を統合した実行可能ファイル
- CLI経由で実行: `python src/core/main_extractor.py input.txt --pilot`

### 📊 評価手法（src/evaluation/）

**azure_perplexity.py**
- Azure OpenAI GPT-4o使用
- 数学的に正確なPerplexity計算（logprobs API）
- マスク予測評価

**enhanced_perplexity.py**  
- 文書のホットスポット検出
- 段階的チャンク分析（500/100/20トークン）
- 複数手法の重み付き平均

**input_logprobs_calculator.py**
- 入力テキストのログ確率計算
- トークン単位での確率分析

**mask_generator.py**
- 評価用のMASK文脈生成
- 4つの生成戦略（実文脈/テンプレート/ドメイン特化/統計的）

### 🔍 抽出手法（src/extraction/）

**generic_perplexity_extractor.py**
- 汎用的なperplexityベースの用語抽出
- 統計的手法による専門用語スコアリング
- Gemini-2.0使用可能

### 📝 サンプル実装（examples/）

**statistical_extractor.py**
- 統計的手法による専門用語抽出
- TF-IDF、C-value、形態素解析を組み合わせ
- オプションでLLM検証も実行

**llm_extractor.py**
- LLMのみを使用した専門用語抽出
- Gemini-2.0による直接抽出
- 難易度・分野・理由付きの構造化出力

### 📁 その他のディレクトリ

**tests/test_data/**
- 各種テストデータ（JSON形式）
- 評価結果やトークン化結果を保存

**logs/**
- 実行ログの保存先
- デバッグ情報の記録

**input/**
- 処理対象文書の配置場所
- 主にPDF文書を想定

**output/**
- 抽出結果の出力先
- JSON形式で結果を保存

**old/**
- 以前のバージョンのコード
- 参考実装として保存

## 実行方法

### メインシステムの実行
```bash
# 基本実行
python src/core/main.py

# 専門用語抽出（メインエクストラクタ）
python src/core/main_extractor.py input.txt --pilot

# 制限モード（上位N語）
python src/core/main_extractor.py input.txt --limit 50

# フルモード（上位100語）
python src/core/main_extractor.py input.txt --full
```

### 評価ツールの実行
```bash
# Azure Perplexityテスト
python src/evaluation/azure_perplexity.py

# 入力ログ確率計算
python src/evaluation/input_logprobs_calculator.py

# MASK生成テスト
python src/evaluation/mask_generator.py
```

### 抽出ツールの実行
```bash
# 汎用perplexity抽出
python src/extraction/generic_perplexity_extractor.py ./input ./output/generic_results.json
```

### サンプルの実行
```bash
# 統計的手法による抽出
python examples/statistical_extractor.py ./input ./output/statistical_terms.json

# LLMのみによる抽出
python examples/llm_extractor.py ./input ./output/llm_terms.json
```

## 主要な評価手法の使い分け

### 1. 高速・簡易評価が必要な場合
→ `main_extractor.py`の統計指標（TF-IDF、C-value）のみ使用

### 2. 高精度評価が必要な場合
→ `main_extractor.py`でAzure OpenAI評価を有効化

### 3. 文書全体の難易度分析
→ `enhanced_perplexity.py`のホットスポット検出

### 4. Perplexityベースの抽出
→ `generic_perplexity_extractor.py`を使用

## 環境設定（.env）

```env
# OpenAI API（オプション）
OPENAI_API_KEY=your_openai_api_key_here

# Google API（Gemini用）
GOOGLE_API_KEY=your_google_api_key_here

# Azure OpenAI（GPT-4o用）
AZURE_OPENAI_API_KEY=your_key
AZURE_OPENAI_ENDPOINT=your_endpoint
AZURE_OPENAI_API_VERSION=2024-12-01-preview
AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4o
AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME=text-embedding-3-small

# Database（オプション）
DATABASE_URL=postgresql://user:password@localhost/jargon_db

# LangSmith（オプション）
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_key
LANGCHAIN_PROJECT=term-extraction
```

## 依存関係

主要なパッケージ：
- PyMuPDF: PDF処理
- transformers: 自然言語処理
- langchain: LLM統合
- sudachipy: 日本語形態素解析
- scikit-learn: 機械学習・クラスタリング
- fastapi: REST API

詳細は`requirements.txt`を参照。