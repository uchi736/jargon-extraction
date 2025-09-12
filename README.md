# Jargon - 専門用語抽出システム

## 概要
Jargonは、PDF、Word、Markdownなどの文書から専門用語を自動的に抽出し、用語辞書を構築するPythonベースのシステムです。自然言語処理技術とLLM（大規模言語モデル）を活用して、高精度な専門用語の抽出と定義生成を行います。

## 主な機能
- 📄 複数形式の文書処理（PDF、DOCX、Markdown、HTML、TXT）
- 🤖 Transformersベースの専門用語抽出
- 🔍 LLM（Gemini/GPT）を使用した用語定義の自動生成
- 📊 用語のクラスタリングと類似度分析
- 💾 PostgreSQL + pgvectorによるベクトル検索対応
- 🚀 FastAPIによるREST API提供
- 📈 リッチなコンソール出力とロギング
- 🎯 **C値・NC値による複合語重要度計算（新機能）**
- ⚡ **Sudachi + Embedding/LLMハイブリッド分かち書き（新機能）**
- 📚 **法令・技術文書特化の用語抽出（新機能）**

## 必要要件
- Python 3.8以上
- PostgreSQL（pgvector拡張付き）
- OpenAI APIキー（LLM機能を使用する場合）

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

# Google Gemini API（必須）
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

### 基本的な実行
```bash
python src/core/main.py
```

### 文書から専門用語を抽出
```bash
python src/core/main_extractor.py --input input/document.pdf --output output/dictionary.json
```

### Perplexityベースの専門用語抽出
```bash
# 汎用perplexity抽出器
python src/extraction/generic_perplexity_extractor.py ./input ./output/dictionary.json
```

### 評価ツールの実行
```bash
# Azure OpenAIによるperplexity計算
python src/evaluation/azure_perplexity.py

# 入力ログ確率の計算
python src/evaluation/input_logprobs_calculator.py
```

### サンプルコードの実行
```bash
# 統計的手法による抽出（TF-IDF + 形態素解析）
python examples/statistical_extractor.py ./input ./output/statistical_terms.json

# LLMのみによる抽出（Gemini-2.0）
python examples/llm_extractor.py ./input ./output/llm_terms.json
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
│   │   └── mask_generator.py          # マスク生成
│   ├── extraction/            # 抽出アルゴリズム
│   │   └── generic_perplexity_extractor.py # 汎用perplexity抽出
│   └── utils/                # ユーティリティ
│       ├── document_loader.py  # 共通文書ローダー
│       └── base_extractor.py   # 抽出器基底クラス
├── tests/                     # テスト関連
│   └── test_data/            # テストデータ
│       ├── final_test.json
│       ├── fixed_results.json
│       ├── improved_results.json
│       ├── legal_text_tokenization.json
│       └── test_output.json
├── examples/                  # サンプル実装
│   ├── statistical_extractor.py # 統計的手法による抽出
│   └── llm_extractor.py        # LLMのみによる抽出
├── logs/                     # ログファイル
│   └── term_extraction.log  # 抽出処理ログ
├── input/                    # 入力文書ディレクトリ
│   └── *.pdf               # 処理対象のPDF文書
├── output/                   # 出力ディレクトリ
│   └── generic_results.json # 生成された抽出結果
├── docs/                     # ドキュメント
│   ├── azure_perplexity_detailed.md
│   ├── evaluation_logic_spec.md
│   └── FILE_STRUCTURE.md
├── old/                      # アーカイブ済みコード
│   └── rag_extractor.py
├── config.yml               # 設定ファイル
├── requirements.txt         # Python依存パッケージ
├── .env                     # 環境変数  
├── .gitignore              # Git除外設定
├── logprobs_calculation_logic.md  # ログ確率計算ロジックの説明
├── 計画書.md                # プロジェクト計画書
└── README.md               # このファイル
```

## 設定
`config.yml`で以下の項目を設定可能：
- 抽出する用語の最小/最大文字数
- クラスタリングのパラメータ
- LLMのモデル選択
- ログレベル
- 出力フォーマット

## API エンドポイント
FastAPIサーバーを起動後、以下のエンドポイントが利用可能：
- `POST /extract` - 文書から用語を抽出
- `GET /terms` - 抽出済み用語一覧を取得
- `POST /search` - ベクトル類似度検索
- `GET /docs` - APIドキュメント（Swagger UI）

## 開発

### テストの実行
```bash
pytest tests/
```

### コードフォーマット
```bash
black .
flake8 .
```

## 技術的詳細

### アーキテクチャ

#### システム全体フロー

```mermaid
flowchart TB
    subgraph Input["📄 入力層"]
        A[PDF文書]
        B[Word文書]
        C[Markdown文書]
        D[HTML文書]
    end
    
    subgraph Parser["🔧 パーサー層"]
        E[PyMuPDF]
        F[python-docx]
        G[Markdown Parser]
        H[BeautifulSoup4]
    end
    
    subgraph Preprocess["🔄 前処理層"]
        I[テキスト正規化]
        J[文分割]
        K[ノイズ除去]
    end
    
    subgraph Extraction["⚙️ 抽出層"]
        L[ルールベース抽出<br/>- 正規表現<br/>- パターンマッチング]
        M[MLベース抽出<br/>- Transformers<br/>- BERT/RoBERTa]
    end
    
    subgraph Processing["🧠 処理層"]
        N[候補フィルタリング]
        O[スコアリング]
        P[重複除去]
    end
    
    subgraph Enhancement["✨ 拡張層"]
        Q[LLM処理<br/>- GPT-4/3.5<br/>- 定義生成]
        R[Embedding生成<br/>- text-embedding-ada-002<br/>- Sentence Transformers]
    end
    
    subgraph Storage["💾 永続化層"]
        S[(PostgreSQL<br/>+ pgvector)]
        T[(Redis Cache)]
    end
    
    subgraph Output["📤 出力層"]
        U[JSON出力]
        V[REST API]
        W[Web UI]
    end
    
    A --> E
    B --> F
    C --> G
    D --> H
    
    E --> I
    F --> I
    G --> I
    H --> I
    
    I --> J
    J --> K
    K --> L
    K --> M
    
    L --> N
    M --> N
    N --> O
    O --> P
    
    P --> Q
    P --> R
    
    Q --> S
    R --> S
    S --> T
    
    S --> U
    S --> V
    V --> W
    
    style Input fill:#e1f5fe
    style Parser fill:#fff3e0
    style Preprocess fill:#f3e5f5
    style Extraction fill:#e8f5e9
    style Processing fill:#fce4ec
    style Enhancement fill:#fff9c4
    style Storage fill:#e0f2f1
    style Output fill:#f1f8e9
```

#### データフロー詳細

```mermaid
sequenceDiagram
    participant U as ユーザー
    participant API as FastAPI
    participant P as Parser
    participant E as Extractor
    participant LLM as OpenAI API
    participant DB as PostgreSQL
    participant C as Cache
    
    U->>API: 文書アップロード
    API->>P: 文書解析要求
    P->>P: 形式判定・変換
    P-->>API: テキストデータ
    
    API->>E: 専門用語抽出
    E->>E: トークン化
    E->>E: 候補生成
    E-->>API: 用語候補リスト
    
    API->>LLM: 定義生成要求
    LLM->>LLM: プロンプト処理
    LLM-->>API: 用語定義
    
    API->>LLM: Embedding生成
    LLM-->>API: ベクトルデータ
    
    API->>DB: データ保存
    DB->>DB: インデックス更新
    DB-->>API: 保存完了
    
    API->>C: キャッシュ更新
    C-->>API: OK
    
    API-->>U: 処理結果
```

#### クラス図

```mermaid
classDiagram
    class DocumentProcessor {
        +process_document(file_path)
        +extract_text()
        +normalize_text()
    }
    
    class TermExtractor {
        <<abstract>>
        +extract_terms()
    }
    
    class RuleBasedExtractor {
        +patterns: List
        +extract_terms()
        +apply_patterns()
    }
    
    class MLBasedExtractor {
        +model: TransformerModel
        +extract_terms()
        +predict_terms()
    }
    
    class LLMProcessor {
        +api_key: str
        +model: str
        +generate_definition()
        +generate_embedding()
    }
    
    class VectorDB {
        +connection: Connection
        +insert_vector()
        +search_similar()
        +update_index()
    }
    
    class TermCandidate {
        +term: str
        +score: float
        +context: str
        +definition: str
        +embedding: Vector
    }
    
    class APIEndpoint {
        +upload_document()
        +extract_terms()
        +search_terms()
        +get_statistics()
    }
    
    DocumentProcessor --> TermExtractor
    TermExtractor <|-- RuleBasedExtractor
    TermExtractor <|-- MLBasedExtractor
    
    RuleBasedExtractor --> TermCandidate
    MLBasedExtractor --> TermCandidate
    
    TermCandidate --> LLMProcessor
    LLMProcessor --> VectorDB
    
    APIEndpoint --> DocumentProcessor
    APIEndpoint --> VectorDB
    APIEndpoint --> TermCandidate
```

#### 状態遷移図

```mermaid
stateDiagram-v2
    [*] --> 待機中
    
    待機中 --> 文書受信: アップロード
    
    文書受信 --> 解析中: パース開始
    解析中 --> 前処理中: テキスト抽出完了
    前処理中 --> 抽出中: 正規化完了
    
    抽出中 --> ルール処理: ルールベース
    抽出中 --> ML処理: MLベース
    
    ルール処理 --> 候補生成
    ML処理 --> 候補生成
    
    候補生成 --> フィルタリング: 候補あり
    候補生成 --> エラー: 候補なし
    
    フィルタリング --> LLM処理: 閾値以上
    フィルタリング --> エラー: 閾値未満
    
    LLM処理 --> Embedding生成: 定義生成完了
    Embedding生成 --> DB保存: ベクトル生成完了
    
    DB保存 --> キャッシュ更新: 保存成功
    DB保存 --> エラー: 保存失敗
    
    キャッシュ更新 --> 完了
    
    完了 --> [*]
    エラー --> [*]
    
    note right of LLM処理
        GPT-4/3.5による
        定義とコンテキスト生成
    end note
    
    note left of ML処理
        BERT/RoBERTaによる
        系列ラベリング
    end note
```

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

##### Transformersベース
- **Hugging Face Transformers**: 事前学習済みモデルの活用
  - BERT日本語モデル（cl-tohoku/bert-base-japanese）
  - RoBERTa日本語モデル（nlp-waseda/roberta-base-japanese）
- **トークナイザー**: 
  - SentencePiece（サブワード分割）
  - BPE（Byte Pair Encoding）

##### 専門用語抽出アルゴリズム
1. **統計的手法**
   - TF-IDF（Term Frequency-Inverse Document Frequency）
   - **C-Value/NC-Value（改良実装済み）**
     - 複合語の統計的重要度を計算
     - 文脈情報を考慮した重み付け
   - 出現頻度ベースのフィルタリング

2. **言語学的手法**
   - 品詞パターンマッチング（名詞句抽出）
   - **複合語解析（強化版）**
     - SudachiPyのA/B/Cモード活用
     - 品詞細分類による結合判定
     - 法令・専門用語パターン辞書
   - 専門用語の形態的特徴抽出

3. **機械学習手法**
   - CRF（Conditional Random Fields）
   - BiLSTM-CRF
   - Transformerベースの系列ラベリング
   
4. **ハイブリッド手法（新規実装）**
   - **Sudachi + Embedding**
     - 文脈ベクトルによる最適粒度選択
     - 境界信頼度スコアリング
     - Azure OpenAI text-embedding-3-small対応
   - **Sudachi + LLM**
     - Gemini 2.0による文脈理解
     - 曖昧性の高い箇所のみLLM判定
     - 法令文書特化プロンプト

#### LLM統合
- **OpenAI API**
  - GPT-4: 高精度な定義生成
  - GPT-3.5-turbo: コスト効率的な処理
  - Function Calling: 構造化出力
- **LangChain**: 
  - プロンプトテンプレート管理
  - チェーン構築（LCEL）
  - メモリ管理
  - ドキュメントローダー

#### ベクトル処理
- **text-embedding-ada-002**: OpenAIの埋め込みモデル
- **Sentence Transformers**: ローカル埋め込み生成
- **次元数**: 1536次元（ada-002）/ 768次元（BERT）

#### データベース
- **PostgreSQL + pgvector**
  - ベクトル類似度検索（コサイン類似度、L2距離）
  - インデックス: IVFFlat、HNSW
  - ハイブリッド検索（キーワード + ベクトル）

#### クラスタリング
- **scikit-learn**
  - K-means: 用語のグループ化
  - DBSCAN: 密度ベースクラスタリング
  - 階層的クラスタリング
- **次元削減**
  - PCA（主成分分析）
  - t-SNE（可視化用）
  - UMAP（高速次元削減）

### パフォーマンス最適化

#### 処理パイプライン最適化

```mermaid
graph LR
    subgraph 従来の処理["🐌 従来の処理"]
        A1[文書1] --> B1[処理]
        B1 --> C1[文書2]
        C1 --> D1[処理]
        D1 --> E1[文書3]
        E1 --> F1[処理]
    end
    
    subgraph 最適化後["🚀 最適化後"]
        A2[文書1] --> B2[処理]
        A3[文書2] --> B3[処理]
        A4[文書3] --> B4[処理]
        B2 --> G[集約]
        B3 --> G
        B4 --> G
    end
    
    style A1 fill:#ffcdd2
    style C1 fill:#ffcdd2
    style E1 fill:#ffcdd2
    style A2 fill:#c8e6c9
    style A3 fill:#c8e6c9
    style A4 fill:#c8e6c9
```

#### キャッシュ戦略

```mermaid
flowchart TD
    Request[リクエスト] --> CheckL1{L1キャッシュ<br/>確認}
    
    CheckL1 -->|ヒット| ReturnL1[即座に返却<br/>〜1ms]
    CheckL1 -->|ミス| CheckL2{L2キャッシュ<br/>確認}
    
    CheckL2 -->|ヒット| ReturnL2[Redis返却<br/>〜10ms]
    CheckL2 -->|ミス| CheckDB{DB確認}
    
    CheckDB -->|存在| ReturnDB[DB返却<br/>〜100ms]
    CheckDB -->|なし| Process[新規処理<br/>〜1000ms]
    
    ReturnL2 --> UpdateL1[L1更新]
    ReturnDB --> UpdateL2[L2更新]
    ReturnDB --> UpdateL1
    Process --> UpdateDB[DB保存]
    UpdateDB --> UpdateL2
    UpdateDB --> UpdateL1
    
    style ReturnL1 fill:#4caf50
    style ReturnL2 fill:#8bc34a
    style ReturnDB fill:#ffc107
    style Process fill:#ff9800
```

#### メモリ管理
- **バッチ処理**: 大規模文書の分割処理
- **ストリーミング**: メモリ効率的な逐次処理
- **キャッシング**: 
  - LRUキャッシュ（頻繁にアクセスされる用語）
  - Redis（分散キャッシュ）

#### 並列処理
- **マルチプロセシング**: CPU集約的タスク
- **非同期処理**: I/O集約的タスク
- **バッチ推論**: GPUの効率的利用

#### 最適化テクニック
```python
# 例: バッチ処理による効率化
async def process_documents_batch(docs, batch_size=10):
    results = []
    for i in range(0, len(docs), batch_size):
        batch = docs[i:i+batch_size]
        batch_results = await asyncio.gather(
            *[process_single_doc(doc) for doc in batch]
        )
        results.extend(batch_results)
    return results
```

#### リソース使用状況

```mermaid
gantt
    title リソース使用タイムライン
    dateFormat HH:mm:ss
    axisFormat %S
    
    section CPU
    文書解析        :active, cpu1, 00:00:00, 3s
    テキスト処理    :active, cpu2, 00:00:02, 4s
    用語抽出        :active, cpu3, 00:00:04, 5s
    
    section GPU
    Embedding生成   :crit, gpu1, 00:00:02, 3s
    モデル推論      :crit, gpu2, 00:00:04, 4s
    
    section I/O
    ファイル読込    :io1, 00:00:00, 1s
    DB書込         :io2, 00:00:08, 2s
    キャッシュ更新  :io3, 00:00:09, 1s
    
    section API
    OpenAI呼出     :api1, 00:00:05, 3s
    レスポンス待機  :api2, 00:00:06, 2s
```

### セキュリティ

#### APIキー管理
- 環境変数による管理（.envファイル）
- シークレット管理ツール対応
- キーローテーション推奨

#### 入力検証
- ファイルタイプ検証
- サイズ制限（デフォルト: 50MB）
- コンテンツスキャニング

#### レート制限
- API呼び出し制限
- 同時接続数制限
- トークンバケットアルゴリズム

### 監視とログ

#### ログシステム
- **loguru**: 構造化ログ
- ログレベル: DEBUG, INFO, WARNING, ERROR, CRITICAL
- ローテーション設定
- 外部ログ収集システム連携（ELK Stack対応）

#### メトリクス
- 処理時間測定
- API呼び出し回数
- エラー率追跡
- リソース使用状況（CPU、メモリ、GPU）

### 拡張性

#### プラグインアーキテクチャ

```mermaid
graph TB
    subgraph Core["🎯 コアシステム"]
        A[Plugin Manager]
        B[Base Interfaces]
        C[Event System]
    end
    
    subgraph Plugins["🔌 プラグイン"]
        D[カスタム抽出器]
        E[新形式パーサー]
        F[外部API連携]
        G[カスタムフィルター]
    end
    
    subgraph Extensions["🎨 拡張機能"]
        H[Slack通知]
        I[Teams連携]
        J[S3ストレージ]
        K[Azure Blob]
    end
    
    B --> D
    B --> E
    B --> F
    B --> G
    
    C --> H
    C --> I
    A --> J
    A --> K
    
    style Core fill:#e3f2fd
    style Plugins fill:#f3e5f5
    style Extensions fill:#e8f5e9
```

#### スケーリングアーキテクチャ

```mermaid
flowchart TB
    subgraph LB["ロードバランサー"]
        nginx[Nginx/HAProxy]
    end
    
    subgraph Workers["ワーカーノード"]
        W1[Worker 1<br/>FastAPI]
        W2[Worker 2<br/>FastAPI]
        W3[Worker 3<br/>FastAPI]
    end
    
    subgraph Queue["キューシステム"]
        RQ[RabbitMQ/Redis Queue]
    end
    
    subgraph Tasks["タスクワーカー"]
        T1[Celery Worker 1]
        T2[Celery Worker 2]
        T3[Celery Worker 3]
    end
    
    subgraph Storage["ストレージ層"]
        DB[(PostgreSQL<br/>Primary)]
        DB2[(PostgreSQL<br/>Replica)]
        Cache[(Redis Cluster)]
        S3[Object Storage]
    end
    
    nginx --> W1
    nginx --> W2
    nginx --> W3
    
    W1 --> RQ
    W2 --> RQ
    W3 --> RQ
    
    RQ --> T1
    RQ --> T2
    RQ --> T3
    
    T1 --> DB
    T2 --> DB
    T3 --> DB
    
    DB --> DB2
    
    W1 --> Cache
    W2 --> Cache
    W3 --> Cache
    
    T1 --> S3
    T2 --> S3
    T3 --> S3
    
    style LB fill:#ffebee
    style Workers fill:#e8eaf6
    style Queue fill:#fff3e0
    style Tasks fill:#e0f2f1
    style Storage fill:#f1f8e4
```

- カスタム抽出器の追加
- 新しい文書形式のサポート
- 外部サービス統合
- 水平スケーリング対応
- ロードバランシング
- キューシステム（Celery、RabbitMQ）

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