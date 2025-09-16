# Azure OpenAI logprobs を使った Perplexity・トークン確率計算ロジック

## 1. 基本概念

### Perplexity とは？
- **定義**: モデルがテキストを予測する際の「困惑度」
- **数式**: `Perplexity = exp(-平均log確率)`
- **意味**: 値が低いほどモデルがよく理解している

### logprobs とは？
- **定義**: 各トークンの対数確率（log probability）
- **範囲**: 負の値（0に近いほど確率が高い）
- **変換**: `確率 = exp(logprob)`

---

## 2. 具体的な計算例：「理論空燃比」

### Step 1: 文脈での予測

```python
# 3つの文脈パターンで「理論空燃比」を予測させる
contexts = [
    "この",           # → 次に「理論空燃比」が来る確率は？
    "理論空燃比について",  # → 「説明します」が続く
    "重要な概念である"    # → 次に「理論空燃比」が来る確率は？
]
```

### Step 2: API呼び出しと logprobs 取得

```python
# Azure OpenAI API 呼び出し
response = await azure_client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": "次の文を自然に継続してください。"},
        {"role": "user", "content": "この"}
    ],
    max_tokens=10,
    temperature=0.0,
    logprobs=True,      # ← 重要！
    top_logprobs=5       # 上位5候補も取得
)
```

### Step 3: レスポンスの構造

```json
{
  "choices": [{
    "logprobs": {
      "content": [
        {
          "token": "理論",
          "logprob": -2.3,     // log(0.1) ≈ -2.3
          "top_logprobs": [
            {"token": "理論", "logprob": -2.3},
            {"token": "技術", "logprob": -3.5},
            {"token": "方法", "logprob": -4.1}
          ]
        },
        {
          "token": "空燃",
          "logprob": -0.5,     // log(0.6) ≈ -0.5
          "top_logprobs": [
            {"token": "空燃", "logprob": -0.5},
            {"token": "的な", "logprob": -2.8}
          ]
        },
        {
          "token": "比",
          "logprob": -0.1,     // log(0.9) ≈ -0.1
          "top_logprobs": [
            {"token": "比", "logprob": -0.1},
            {"token": "値", "logprob": -3.2}
          ]
        }
      ]
    }
  }]
}
```

### Step 4: Perplexity 計算

```python
# 各トークンの logprob を収集
token_logprobs = [-2.3, -0.5, -0.1]

# 平均 logprob を計算
avg_logprob = sum(token_logprobs) / len(token_logprobs)
# = (-2.3 + -0.5 + -0.1) / 3
# = -2.9 / 3
# = -0.967

# Perplexity を計算
perplexity = exp(-avg_logprob)
# = exp(-(-0.967))
# = exp(0.967)
# = 2.63
```

**結果**: Perplexity = 2.63（非常に低い = よく理解されている）

---

## 3. 実際の計算例（3つの文脈での平均）

### 文脈1: 「この」→「理論空燃比」

```python
logprobs_1 = [-2.3, -0.5, -0.1]  # 理論, 空燃, 比
avg_logprob_1 = -0.967
perplexity_1 = 2.63
```

### 文脈2: 「理論空燃比について」→「説明します」

```python
logprobs_2 = [-0.2, -0.3]  # 説明, します
avg_logprob_2 = -0.25
perplexity_2 = 1.28
```

### 文脈3: 「重要な概念である」→「理論空燃比」

```python
logprobs_3 = [-3.5, -1.2, -0.3]  # 理論, 空燃, 比
avg_logprob_3 = -1.67
perplexity_3 = 5.31
```

### 最終 Perplexity

```python
all_perplexities = [2.63, 1.28, 5.31]
final_perplexity = mean(all_perplexities)
# = (2.63 + 1.28 + 5.31) / 3
# = 3.07

# 信頼度分類
if final_perplexity < 20:
    confidence = "high_confidence"  # ← この場合
elif final_perplexity < 50:
    confidence = "medium_confidence"
else:
    confidence = "low_confidence"
```

---

## 4. マスク予測方式の例

### 入力テキスト
```
「アンモニア燃料の[MASK]を正確に計算する」
```

### API呼び出し
```python
response = await azure_client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "user", "content": "次の[MASK]に入る専門用語は？\nアンモニア燃料の[MASK]を正確に計算する"}
    ],
    max_tokens=5,
    logprobs=True,
    top_logprobs=10  # 上位10候補
)
```

### レスポンス分析
```python
top_predictions = [
    {"token": "効率", "logprob": -0.8, "prob": 0.45},
    {"token": "濃度", "logprob": -1.2, "prob": 0.30},
    {"token": "理論空燃比", "logprob": -2.5, "prob": 0.08},  # ← 正解
    {"token": "特性", "logprob": -3.1, "prob": 0.04}
]

# 正解の順位とlogprobから Perplexity 計算
if "理論空燃比" in predictions:
    rank = 3  # 3位
    logprob = -2.5
    perplexity = exp(-logprob) = exp(2.5) = 12.18
```

---

## 5. 比較例：専門用語 vs 一般語

### 専門用語「理論空燃比」
```python
contexts = ["この", "について", "を計算"]
logprobs = [-2.3, -0.5, -0.1]  # 予測が困難
avg = -0.967
perplexity = 2.63  # 低い（理解されている）
```

### 未知の専門用語「ポリアセチルモルフォリン」
```python
contexts = ["この", "について", "を計算"]
logprobs = [-8.5, -7.2, -9.1]  # 予測が非常に困難
avg = -8.27
perplexity = 3924.5  # 非常に高い（理解されていない）
```

### 一般語「データ」
```python
contexts = ["この", "について", "を計算"]
logprobs = [-0.5, -0.3, -0.2]  # 予測が容易
avg = -0.33
perplexity = 1.39  # 非常に低い（よく理解されている）
```

---

## 6. 優先度スコアの計算

### 総合評価式
```python
priority_score = (
    (100 - perplexity) * 0.4 +        # Perplexity (40%)
    token_probability * 100 * 0.2 +    # トークン確率 (20%)
    statistical_score * 0.4             # 統計スコア (40%)
)
```

### 実例：「理論空燃比」
```python
perplexity = 29.4
token_probability = 0.09  # 9%
statistical_score = 3.76  # C-value + TF-IDF

priority_score = (
    (100 - 29.4) * 0.4 +     # = 28.24
    0.09 * 100 * 0.2 +       # = 1.8
    3.76 * 0.4               # = 1.504
)
# = 31.54
```

---

## 7. 文書全体のホットスポット検出

### チャンク分割
```python
text = "理論空燃比は内燃機関において重要な指標である。アンモニア燃料の理論空燃比を..."
chunks = split_by_words(text, chunk_size=50, overlap=0.5)
```

### 各チャンクの Perplexity
```python
chunk_perplexities = [
    {"chunk": 1, "text": "理論空燃比は内燃機関...", "perplexity": 15.2},
    {"chunk": 2, "text": "アンモニア燃料の理論...", "perplexity": 18.7},
    {"chunk": 3, "text": "ポリアセチルモルフォリン...", "perplexity": 85.3}  # ← ホットスポット！
]

# ホットスポット判定
mean = 39.7
std = 30.8
threshold = mean + 1.5 * std = 85.9

hotspots = [chunk for chunk in chunks if chunk["perplexity"] > threshold]
```

---

## 8. まとめ：なぜ logprobs が優れているか

### 従来の方法（間接的）
```python
# 3回定義を生成して一貫性を見る
definitions = [
    "理論空燃比は燃料と空気の理想的な比率",
    "理論空燃比とは完全燃焼に必要な空気量の比",
    "理論空燃比は化学量論的な空燃比"
]
# → 定義のばらつきから「推定」
perplexity_estimate = 30-40
```

### logprobs 方式（直接的）
```python
# モデルの内部確率を直接取得
logprobs = [-0.967, -1.2, -0.8]
# → 数学的に正確な計算
perplexity_exact = exp(-mean(logprobs)) = 3.07
```

### 利点
1. **正確性**: 推定ではなく真の確率
2. **高速**: 1回の API 呼び出しで完了
3. **一貫性**: 同じ入力なら同じ結果
4. **詳細情報**: 各トークンの確率分布も取得可能

---

## 9. 実装のベストプラクティス

### 1. 複数文脈での評価
```python
# 悪い例：単一文脈
perplexity = calculate_perplexity("この" + term)

# 良い例：複数文脈の平均
contexts = [
    f"この{term}は",
    f"{term}について",
    f"重要な{term}を"
]
perplexities = [calculate_perplexity(c) for c in contexts]
final_perplexity = mean(perplexities)
```

### 2. エラー処理
```python
if response.choices[0].logprobs is None:
    # logprobs が利用不可の場合はフォールバック
    return fallback_perplexity_calculation()
```

### 3. 閾値の調整
```python
# ドメインに応じて調整
THRESHOLDS = {
    "technical": {"high": 30, "medium": 70},
    "general": {"high": 20, "medium": 50},
    "academic": {"high": 25, "medium": 60}
}
```

---

## 10. トラブルシューティング

### 問題: Perplexity が異常に低い（< 2）
**原因**: 一般的すぎる語
**対策**: 統計スコアとの組み合わせで調整

### 問題: logprobs が None
**原因**: モデルが非対応 or API バージョンが古い
**対策**: 
```python
api_version="2024-12-01-preview"  # 最新版を使用
model="gpt-4o"  # logprobs 対応モデル
```

### 問題: API コストが高い
**対策**: 統計的前処理で候補を絞る
```python
# 上位100語のみ LLM 評価
candidates = statistical_ranking(all_terms)[:100]
```