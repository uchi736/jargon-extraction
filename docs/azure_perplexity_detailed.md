# Azure Perplexity 詳細解説

## 📊 Perplexityとは？

Perplexity（困惑度）は、言語モデルがテキストをどれだけ「理解しやすい」と判断するかを示す指標です。

- **低いPerplexity（< 30）** = モデルが理解しやすい = 一般的な専門用語
- **高いPerplexity（> 70）** = モデルが理解しにくい = 特殊・難解な専門用語

## 🔧 3つの計算手法

### 1️⃣ 真のPerplexity計算（logprobsベース）

最も数学的に正確な手法です。

#### 仕組み
```python
# ステップ1: 文脈を作成
"理論空燃比は内燃機関において" → [ここでモデルに続きを予測させる]

# ステップ2: モデルが各トークンを生成する確率（logprob）を取得
トークン "重要" → logprob: -0.5 (確率: 60%)
トークン "な" → logprob: -0.3 (確率: 74%)
トークン "指標" → logprob: -1.2 (確率: 30%)

# ステップ3: Perplexity計算
avg_logprob = (-0.5 + -0.3 + -1.2) / 3 = -0.67
perplexity = exp(-avg_logprob) = exp(0.67) = 1.95
```

#### 具体例：「理論空燃比」の評価

```python
# 実際のテキストから文脈を抽出
text = "理論空燃比は内燃機関において重要な指標である。"

# 3つの文脈パターンを作成
contexts = [
    "理論空燃比は",           # prefix（用語の前まで）
    "において理論空燃比を",     # 文中での使用
    "重要な理論空燃比"         # 修飾付き
]

# 各文脈でのPerplexityを計算し平均化
結果:
- 文脈1: Perplexity = 15.3 (モデルが予測しやすい)
- 文脈2: Perplexity = 18.7
- 文脈3: Perplexity = 12.1
→ 平均Perplexity = 15.4 (高確信度！)
```

### 2️⃣ マスク予測方式

用語を[MASK]に置換して、モデルが正しく予測できるかをテストします。

#### 仕組み
```python
# 元の文
"理論空燃比は内燃機関において重要な指標である。"
↓
# マスク化
"[MASK]は内燃機関において重要な指標である。"
↓
# モデルに予測させる
"[MASK]に入る専門用語を予測してください"
```

#### 評価基準

| 予測結果 | 順位 | Perplexity | 解釈 |
|---------|------|------------|------|
| 完全一致 | 1位 | 5-20 | 非常に理解しやすい |
| 完全一致 | 2-3位 | 20-40 | 理解しやすい |
| 部分一致 | 4-10位 | 40-70 | やや難解 |
| 不一致 | 圏外 | 70-100 | 非常に難解 |

#### 具体例：3つの用語の比較

```python
用語: "理論空燃比"
マスク文: "[MASK]は内燃機関において重要な指標である。"
モデル予測: "理論空燃比" (1位で正解！)
→ Perplexity = 12.5 (高確信度)

用語: "アンモニア燃料"
マスク文: "[MASK]の理論空燃比を正確に計算する。"
モデル予測: "燃料" (部分的に正解)
→ Perplexity = 45.8 (中確信度)

用語: "NH3-H2混焼"
マスク文: "[MASK]システムの効率を最適化する。"
モデル予測: "エンジン" (不正解)
→ Perplexity = 85.2 (低確信度)
```

### 3️⃣ 文書プロファイル分析

文書全体を細かく分析し、理解困難な箇所（ホットスポット）を検出します。

#### 仕組み
```python
# 100語ごとにチャンク分割（50%オーバーラップ）
チャンク1: 語0-100
チャンク2: 語50-150  # 50語重複
チャンク3: 語100-200
...

# 各チャンクのPerplexity計算
チャンク1: Perplexity = 25.3
チャンク2: Perplexity = 28.7
チャンク3: Perplexity = 68.4 ← ホットスポット！
```

#### ホットスポット検出
```python
平均Perplexity = 30.0
標準偏差 = 15.0
閾値 = 平均 + 1.5 × 標準偏差 = 52.5

チャンク3 (68.4) > 閾値 (52.5) 
→ このチャンクは特に難解な専門用語が集中している！
```

## 📈 実際の処理フロー

### 入力例
```python
text = """
理論空燃比は内燃機関において重要な指標である。
アンモニア燃料の理論空燃比を正確に計算することで、
エンジンの効率を最適化できる。
"""

term = "理論空燃比"
```

### 処理ステップ

1. **文脈抽出**
   ```python
   # 用語の出現箇所を検索
   位置1: 0文字目 "理論空燃比は..."
   位置2: 42文字目 "...アンモニア燃料の理論空燃比を..."
   ```

2. **logprobs取得**
   ```python
   # Azure OpenAI APIコール
   response = await azure_client.chat.completions.create(
       model="gpt-4o",
       messages=[{"role": "user", "content": "理論空燃比は"}],
       logprobs=True,  # ← これが重要！
       top_logprobs=5   # 上位5候補も取得
   )
   ```

3. **トークン分析**
   ```python
   生成されたトークン:
   - "内燃" → logprob: -0.8, 確率: 45%
   - "機関" → logprob: -0.4, 確率: 67%
   - "において" → logprob: -0.2, 確率: 82%
   
   平均logprob = -0.47
   Perplexity = exp(0.47) = 1.6 (非常に低い = 理解しやすい)
   ```

4. **確信度分類**
   ```python
   if perplexity < 20:
       return "high_confidence"  # 自動承認OK
   elif perplexity < 50:
       return "medium_confidence"  # 保留
   else:
       return "low_confidence"  # 人手確認必要
   ```

## 🎯 評価結果の解釈

### 高確信度の例（Perplexity < 20）
```json
{
    "term": "理論空燃比",
    "perplexity": 15.4,
    "confidence": "high_confidence",
    "interpretation": "一般的な専門用語として認識されている"
}
```
→ **自動的に辞書に追加してOK**

### 中確信度の例（20 ≤ Perplexity ≤ 50）
```json
{
    "term": "触媒コンバータ",
    "perplexity": 38.2,
    "confidence": "medium_confidence",
    "interpretation": "専門用語だが文脈により理解度が変わる"
}
```
→ **追加の文脈情報や定義が必要**

### 低確信度の例（Perplexity > 50）
```json
{
    "term": "SCR-DeNOx",
    "perplexity": 78.5,
    "confidence": "low_confidence",
    "interpretation": "特殊な専門用語または造語の可能性"
}
```
→ **専門家による確認が必須**

## 💡 なぜlogprobsが重要か？

通常のLLM評価では「生成されたテキスト」しか見ませんが、logprobsを使うと：

1. **確率的な確信度**が数値で分かる
2. **代替候補**も確認できる（2位、3位の予測も見られる）
3. **数学的に正確**なPerplexity計算が可能

### logprobsなしの場合（従来手法）
```
Q: 「理論空燃比」を知っていますか？
A: はい、知っています。
→ 本当に理解しているか不明
```

### logprobsありの場合（本手法）
```
Q: 「理論空燃比は」の続きを予測
A: 「内燃機関の...」
  - 確率: 67% (logprob: -0.4)
  - 2位候補:「エンジンの...」確率: 20%
  - 3位候補:「燃料の...」確率: 8%
→ 高い確率で正しく予測 = 理解している！
```

## 🚀 実用的な使い方

```python
# 初期化
calculator = AzureLogprobsPerplexityCalculator()

# 単一用語の評価
result = await calculator.calculate_true_perplexity(
    text="理論空燃比は内燃機関において...",
    term="理論空燃比"
)

if result['confidence'] == 'high_confidence':
    print(f"✅ {result['term']}は自動承認可能")
elif result['confidence'] == 'low_confidence':
    print(f"⚠️ {result['term']}は専門家確認が必要")

# 文書全体の難易度分析
profile = await calculator.calculate_document_perplexity_profile(text)
hotspots = [c for c in profile if c['is_hotspot']]
print(f"難解な箇所が{len(hotspots)}箇所見つかりました")
```

## まとめ

Azure Perplexityは、**数学的に正確な確率**に基づいて専門用語の理解度を評価します。これにより：

- 自動承認可能な用語を高精度で選別
- 人手確認が必要な用語を確実に検出
- 文書の難易度プロファイルを可視化

できるため、効率的な専門用語辞書の構築が可能になります。