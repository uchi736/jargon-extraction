#!/usr/bin/env python3
"""
テスト用の簡単な未知語評価デモ
"""

import json
from src.evaluation.unknown_term_evaluator import UnknownTermEvaluator

# テスト用のサンプルデータ
test_data = {
    "metadata": {
        "extraction_date": "2025-09-13",
        "extractor": "LLMTermExtractor"
    },
    "results": [{
        "file_path": "test.pdf",
        "terms": [
            {
                "term": "マイクロパイロット方式",
                "definition": "少量の燃料噴射で着火を促すガスエンジンの着火方式。アンモニアのような着火しにくい燃料に適用される技術。",
                "score": 0.8
            },
            {
                "term": "カーボンニュートラル",
                "definition": "温室効果ガスの排出量と吸収量を均衡させる状態。",
                "score": 0.7
            },
            {
                "term": "API",
                "definition": "アプリケーションプログラミングインターフェース。ソフトウェア間の通信規約。",
                "score": 0.6
            }
        ]
    }]
}

# テストファイルを作成
with open("output/test_terms.json", "w", encoding="utf-8") as f:
    json.dump(test_data, f, ensure_ascii=False, indent=2)

print("Starting unknown term evaluation demo...")
print("=" * 60)

# 評価器を初期化
evaluator = UnknownTermEvaluator(
    similarity_threshold=0.5,  # 低めの閾値でテスト
    llm_model="gemini-1.5-flash"
)

# 各用語を個別に評価してデモ
for result in test_data["results"]:
    for term_data in result["terms"]:
        print(f"\nEvaluating: {term_data['term']}")
        print(f"Original definition: {term_data['definition'][:50]}...")
        
        # コンテキストなし定義を生成
        context_free = evaluator.generate_context_free_definition(term_data['term'])
        print(f"Context-free definition: {context_free[:50]}...")
        
        # 埋め込みと類似度計算
        if context_free:
            orig_embed = evaluator.get_embedding(term_data['definition'])
            free_embed = evaluator.get_embedding(context_free)
            similarity = evaluator.calculate_cosine_similarity(orig_embed, free_embed)
            
            is_unknown = similarity < evaluator.similarity_threshold
            status = "UNKNOWN TERM" if is_unknown else "Known term"
            
            print(f"Similarity: {similarity:.3f}")
            print(f"Status: {status}")
            print("-" * 40)

print("\n" + "=" * 60)
print("Demo completed!")