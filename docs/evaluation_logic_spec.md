# 専門用語抽出・評価システム 評価ロジック仕様書

## システム概要
本システムは、日本語テキストから専門用語を自動抽出し、LLMによる評価を組み合わせて高精度な専門用語辞書を生成するシステムです。

## 主要コンポーネント

### 1. examples/statistical_extractor.py (統計的手法による抽出)
**目的**: 統計的手法とオプションのLLM検証を組み合わせた専門用語抽出

**主要機能**:
- SudachiPy（Mode.A）による形態素解析
- TF-IDFスコア計算
- C-value（複合語重要度）計算
- オプションでGemini-2.0による専門用語検証
- 共通モジュール（document_loader、base_extractor）使用

**処理フロー**:
1. ドキュメント読み込み（共通ローダー使用）
2. 形態素解析で候補語生成（単一名詞・複合名詞）
3. TF-IDFとC-valueでスコアリング
4. 統合スコアで優先順位付け
5. オプションでLLM検証・定義生成

**評価基準**:
- 統計的重要度（TF-IDF）
- 複合語としての専門性（C-value）
- 出現頻度と文脈の豊富さ

### 2. examples/llm_extractor.py (LLMベース抽出)
**目的**: LLMの言語理解力のみに基づく専門用語抽出

**主要機能**:
- Gemini-2.0による直接抽出
- 構造化出力（Pydantic）
- 難易度・分野・理由の付与
- 共通モジュール使用

**処理フロー**:
1. ドキュメント読み込み（共通ローダー使用）
2. テキストをチャンクに分割（4000文字単位）
3. 各チャンクからLLMで専門用語抽出
4. 重複除去と統合
5. 難易度スコアでソート

**評価基準**:
- LLMによる専門性判定
- 難易度評価（1-10スケール）
- 分野特定と理由付け

### 3. src/core/main_extractor.py (統合型評価システム)
**目的**: 統計的指標とLLMによる真のPerplexity計算を組み合わせた高精度評価

**主要機能**:
- SudachiPy（Mode.A）による形態素解析と複合名詞生成
- TF-IDF、C-value/NC-value統計指標計算
- Azure OpenAI GPT-4oによる真のPerplexity計算
- 3段階自動分類（高/中/低確信度）
- 段階的実行モード（pilot/limit/full）

**評価手法**:

#### 3.1 統計的評価
- **TF-IDF**: 文書内での重要度評価
- **C-value**: 専門用語らしさの統計指標
  ```
  C-value = log2(長さ) × (頻度 - 包含頻度/包含数)
  ```

#### 3.2 LLM評価（真のPerplexity）
**logprobs APIベース（利用可能な場合）**:
- 文脈での用語生成確率を直接計算
- 複数文脈での平均logprobからPerplexity算出
- 数式: `Perplexity = exp(-avg_logprob)`

**フォールバック評価（3つの手法の重み付き平均）**:
1. **定義生成の一貫性** (40%)
   - 3回の定義生成で一貫性を評価
   - 不確実性キーワード検出
2. **理解度直接評価** (30%)
   - 1-10スケールで困難度評価
3. **文脈での自然さ** (30%)
   - サンプル文脈での自然さを評価

#### 3.3 トークン生成確率
- 部分完成確率（70%）: 用語の一部から完全形を予測
- 一般性評価（30%）: 専門用語としての一般性を評価

#### 3.4 最終優先度スコア
```
priority_score = 統計スコア × 0.4 + (100 - Perplexity) × 0.4 + トークン確率 × 100 × 0.2
```

### 4. src/evaluation/enhanced_perplexity.py (強化版Perplexity計算)
**目的**: 文脈チャンク化によるホットスポット検出と精密評価

**主要機能**:
- 段階的チャンク分析（500/100/20トークン単位）
- ホットスポット検出（平均+1.5σ以上のPerplexity）
- 複数手法による精密トークン確率計算

**評価手法**:
1. **チャンクPerplexity分析**
   - 異なるサイズでテキストを分割
   - 各チャンクのPerplexityを計算
   - 統計的外れ値をホットスポットとして検出

2. **精密トークン確率（3手法の重み付き平均）**
   - 部分完成確率（40%）
   - 文脈出現確率（40%）
   - 類似語比較確率（20%）

### 5. src/evaluation/azure_perplexity.py (数学的に正確なPerplexity)
**目的**: Azure OpenAI logprobs APIを使用した真のPerplexity計算

**主要機能**:
- logprobsデータの直接取得と解析
- マスク予測方式によるPerplexity計算
- 文書全体のPerplexityプロファイル作成

**評価手法**:

#### 5.1 真のPerplexity計算
```python
# 各トークンのlogprobを取得
logprobs = [token.logprob for token in response.logprobs.content]
# 平均logprobからPerplexity計算
avg_logprob = np.mean(logprobs)
perplexity = math.exp(-avg_logprob)
```

#### 5.2 マスク予測評価
- 用語を[MASK]に置換
- LLMに予測させて正解率を評価
- 予測順位からPerplexityを推定

#### 5.3 文書プロファイル
- 50%オーバーラップでチャンク分割
- 各チャンクのPerplexity計算
- ホットスポット検出（μ + 1.5σ）

### 6. src/evaluation/mask_generator.py (文脈生成戦略)
**目的**: 専門用語評価のための多様なMASK文脈生成

**文脈生成戦略**:
1. **実文脈抽出**: 元テキストから実際の使用例を抽出
2. **テンプレート生成**: 定義/説明/計算/応用の4パターン
3. **ドメイン特化生成**: 工学/化学/AI分野別テンプレート
4. **統計的生成**: 共起パターンに基づく文脈

**品質評価基準**:
- 文脈長（10-200文字が理想）
- MASK位置（文頭・文末を避ける）
- 文法的完全性
- コンテキストの豊富さ

### 7. src/evaluation/input_logprobs_calculator.py (入力ログ確率計算)
**目的**: 入力テキストのトークン単位でのログ確率計算と分析

**主要機能**:
- Azure OpenAI APIを使用したlogprobs取得
- トークン単位での確率分析
- 困惑度の高い箇所の特定

### 8. src/extraction/generic_perplexity_extractor.py (汎用Perplexity抽出器)
**目的**: Perplexityベースの汎用的な専門用語抽出

**主要機能**:
- 統計的手法による専門用語スコアリング
- Gemini-2.0による用語評価
- 段階的な専門用語候補の絞り込み

### 9. src/core/main.py (メインアプリケーション)
**目的**: 文書処理の中核となるアプリケーション

**主要機能**:
- PyMuPDFを使用したPDF処理
- 各種文書フォーマット（PDF、Word、HTML、Markdown）の読み込み
- 非同期処理による効率的な文書処理

### 10. src/utils/document_loader.py (共通文書ローダー)
**目的**: 各種文書形式の統一的な読み込み

**主要機能**:
- PDF、Word、Markdown、HTML、TXTの読み込み
- テキスト分割（チャンク化）
- メタデータ取得
- エンコーディング自動検出

### 11. src/utils/base_extractor.py (抽出器基底クラス)
**目的**: 専門用語抽出器の共通インターフェース

**主要機能**:
- ファイル/ディレクトリ処理の共通化
- 結果の保存（JSON/CSV）
- 表形式での表示
- 統計情報の計算
- 重複除去と統合

## 分類基準

### 高確信度（Perplexity < 30 または スコア > 0.7）
- 自動承認される専門用語
- 明確な定義が生成可能
- 一貫した理解が示される

### 中確信度（30 ≤ Perplexity ≤ 70 または 0.3 ≤ スコア ≤ 0.7）
- 保留状態の用語
- 文脈により理解度が変わる
- 追加検証が推奨される

### 低確信度（Perplexity > 70 または スコア < 0.3）
- 人手確認が必要な用語
- 定義が不明確または不一致
- 専門性が極めて高いか造語の可能性

## 実行モード

### Pilotモード
- 上位10語のみ処理
- 動作確認用
- 高速実行

### Limitモード
- 上位N語を処理（ユーザー指定）
- 部分的な辞書構築
- バランス重視

### Fullモード
- 上位100語を処理
- 完全な辞書構築
- 品質重視

## 出力形式

### results.json
```json
{
  "metadata": {
    "extraction_date": "ISO形式日時",
    "extractor": "使用した抽出器",
    "total_files": 処理ファイル数,
    "total_terms": 総用語数
  },
  "results": [
    {
      "file_path": "ファイルパス",
      "terms": [
        {
          "term": "専門用語",
          "definition": "定義",
          "score": スコア値,
          "frequency": 出現頻度,
          "contexts": ["文脈1", "文脈2"],
          "metadata": {追加情報}
        }
      ]
    }
  ]
}
```

### review.csv
人手確認用の候補リスト（優先度順）
- term: 専門用語
- score: スコア値
- frequency: 出現頻度
- definition: 生成された定義

## 技術的特徴

### 並列処理
- 非同期I/O（asyncio）
- チャンク並列処理
- バッチ推論

### メモリ効率
- ストリーミング処理
- チャンク分割
- 逐次的なガベージコレクション

### エラーハンドリング
- グレースフルデグレード
- フォールバック処理
- 詳細なロギング

## 依存関係

### 必須パッケージ
- PyMuPDF: PDF処理
- sudachipy: 日本語形態素解析
- langchain: LLM統合
- scikit-learn: TF-IDF計算
- pydantic: データモデル定義

### オプションパッケージ
- chromadb: ベクトルDB（RAG用）
- openai: Azure OpenAI API
- google-generativeai: Gemini API