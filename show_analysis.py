#!/usr/bin/env python3
"""
類似度分析結果を表示
"""

import json
import sys

def show_results(json_file="output/similarity_analysis.json"):
    """分析結果を表示"""
    
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print("=" * 60)
    print("類似度分析結果サマリー")
    print("=" * 60)
    
    # 統計情報
    stats = data.get("statistics", {})
    print(f"総用語数: {data.get('total_terms', 0)}")
    print(f"明示的未知語（定義不明）: {stats.get('explicitly_unknown_count', 0)}用語")
    print(f"有効な類似度計算: {stats.get('valid_count', 0)}用語")
    print(f"エラー: {stats.get('error_count', 0)}用語")
    
    # 閾値別カウント
    print("\n閾値別の未知語数:")
    results = data.get("results", [])
    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    unknown_count = sum(1 for r in results if r.get("is_unknown", False))
    
    for threshold in thresholds:
        count = sum(1 for r in results 
                   if (r.get("similarity", 1.0) < threshold or r.get("is_unknown", False)))
        print(f"  < {threshold:.1f}: {count}用語")
    
    # 類似度が低い用語TOP10
    print("\n" + "=" * 60)
    print("類似度が低い用語TOP10（真の未知語候補）")
    print("=" * 60)
    
    # is_unknownフラグ優先でソート
    sorted_results = sorted(results, 
                          key=lambda x: (-1 if x.get("is_unknown") else x.get("similarity", 999)))
    
    for i, result in enumerate(sorted_results[:10]):
        term = result.get("term", "N/A")
        sim = result.get("similarity", -1)
        field = result.get("field", "N/A")
        is_unknown = result.get("is_unknown", False)
        
        if is_unknown:
            status = "[明示的未知語]"
            sim_str = "0.000"
        elif sim >= 0:
            status = ""
            sim_str = f"{sim:.3f}"
        else:
            status = "[エラー]"
            sim_str = "N/A"
        
        print(f"{i+1:2}. {term:20} : {sim_str:5} ({field:10}) {status}")
        
        # 定義の表示（最初の50文字）
        orig_def = result.get("original_definition", "")[:50]
        free_def = result.get("context_free_definition", "")[:50]
        print(f"    元定義: {orig_def}...")
        print(f"    生成定義: {free_def}...")
        print()
    
    # 推奨閾値
    print("=" * 60)
    print("推奨閾値設定")
    print("=" * 60)
    print("0.5: 厳しめ - 真に専門的な用語のみ（約6用語）")
    print("0.6: バランス - 専門性の高い用語（約11用語）")
    print("0.7: 緩め - 一般的でない専門用語全般（約22用語）")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        show_results(sys.argv[1])
    else:
        show_results()