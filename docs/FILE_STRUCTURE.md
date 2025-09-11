# ファイル構成説明書

## 整理後のディレクトリ構造

```
Jargon/
├── src/                        # メインソースコード
│   ├── core/                   # コアシステム
│   │   └── main_extractor.py   # 統合型専門用語抽出システム（メイン）
│   │
│   ├── evaluation/             # 評価手法モジュール
│   │   ├── azure_perplexity.py     # Azure OpenAI logprobs真のPerplexity計算
│   │   ├── enhanced_perplexity.py  # 強化版Perplexity（ホットスポット検出）
│   │   └── mask_generator.py       # MASK文脈生成戦略
│   │
│   ├── extraction/             # 抽出手法モジュール
│   │   └── rag_extractor.py   # RAG統合型抽出（ChromaDB + LCEL）
│   │
│   └── utils/                  # ユーティリティ（今後追加）
│
├── examples/                   # サンプル実装
│   ├── simple_extractor.py    # シンプル版抽出
│   ├── lcel_extractor.py      # LCEL記法版
│   └── embedding_extractor.py # エンベディング版
│
├── docs/                       # ドキュメント
│   ├── evaluation_logic_spec.md  # 評価ロジック仕様書
│   └── FILE_STRUCTURE.md        # 本ファイル
│
├── output/                     # 実行結果出力
│   ├── results.json
│   ├── review.csv
│   └── stats.json
│
├── requirements.txt            # 依存パッケージ
├── .env                        # 環境変数設定
└── README.md                   # プロジェクト説明
```

## 各ファイルの役割

### 🎯 メインシステム（src/core/）
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

**mask_generator.py**
- 評価用のMASK文脈生成
- 4つの生成戦略（実文脈/テンプレート/ドメイン特化/統計的）

### 🔍 抽出手法（src/extraction/）

**rag_extractor.py**
- ChromaDB + LangChain LCEL実装
- ベクトル検索による関連文脈活用
- Gemini-2.0-flash使用

### 📝 サンプル実装（examples/）
過去バージョンや代替実装を保存
- simple_extractor.py: 基本実装
- lcel_extractor.py: LCEL記法版
- embedding_extractor.py: エンベディング重視版

## 実行方法

### メインシステムの実行
```bash
# Pilotモード（10語のみ、動作確認）
python src/core/main_extractor.py input.txt --pilot

# 制限モード（上位N語）
python src/core/main_extractor.py input.txt --limit 50

# フルモード（上位100語）
python src/core/main_extractor.py input.txt --full
```

### 個別モジュールのテスト
```bash
# Azure Perplexityテスト
python src/evaluation/azure_perplexity.py

# MASK生成テスト
python src/evaluation/mask_generator.py

# RAG抽出実行
python src/extraction/rag_extractor.py ./input ./output/dictionary.json
```

## 主要な評価手法の使い分け

### 1. 高速・簡易評価が必要な場合
→ `main_extractor.py`の統計指標（TF-IDF、C-value）のみ使用

### 2. 高精度評価が必要な場合
→ `main_extractor.py`でAzure OpenAI評価を有効化

### 3. 文書全体の難易度分析
→ `enhanced_perplexity.py`のホットスポット検出

### 4. RAGによる文脈考慮が必要な場合
→ `rag_extractor.py`を単独実行

## 環境設定（.env）

```env
# Google API（Gemini用）
GOOGLE_API_KEY=your_key

# Azure OpenAI（GPT-4o用）
AZURE_OPENAI_API_KEY=your_key
AZURE_OPENAI_ENDPOINT=your_endpoint
AZURE_OPENAI_API_VERSION=2024-12-01-preview
AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4o

# LangSmith（オプション）
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_key
LANGCHAIN_PROJECT=term-extraction
```

## 出力ファイル

**output/results.json**
- 3段階分類結果（高/中/低確信度）
- メタデータ（処理時間、候補数）

**output/review.csv**
- 人手確認用リスト
- 優先度順にソート

**output/stats.json**
- 処理統計
- 自動承認率/要確認率