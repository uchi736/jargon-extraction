#!/usr/bin/env python3
"""
全用語の類似度分析スクリプト
閾値調整のために全ての用語の類似度を出力
"""

import json
import time
import os
from pathlib import Path
from typing import Dict, List
from datetime import datetime
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from openai import AzureOpenAI
import numpy as np
from rich.console import Console
from rich.table import Table
from rich.progress import track

load_dotenv()
console = Console()


class SimilarityAnalyzer:
    """類似度分析器"""
    
    def __init__(self, llm_model: str = "gemini-1.5-flash"):
        """初期化"""
        # LLM設定
        self.llm = ChatGoogleGenerativeAI(
            model=llm_model,
            temperature=0.1,
            max_tokens=200
        )
        
        # Azure OpenAI設定
        self.azure_client = AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
        )
        self.embedding_deployment = os.getenv(
            "AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME",
            "text-embedding-3-small"
        )
    
    def generate_definition(self, term: str, field: str = None, retry_count: int = 3) -> str:
        """コンテキストなし定義生成（改良版）"""
        
        # 分野別プロンプト調整
        field_hint = f"（{field}分野）" if field else ""
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """あなたは専門用語辞典の編集者です。
一般的な事前知識の範囲のみで定義してください。
確信が持てない場合は「定義不明」と答えてください。"""),
            ("user", """次の用語{field_hint}を1-2文で定義してください。
推測や曖昧な説明は避け、わからない場合は「定義不明」と答えてください。

用語: {term}""")
        ])
        
        for attempt in range(retry_count):
            try:
                chain = prompt | self.llm
                response = chain.invoke({"term": term, "field_hint": field_hint})
                return response.content.strip()
            except Exception as e:
                if "429" in str(e) and attempt < retry_count - 1:
                    wait_time = (attempt + 1) * 5
                    time.sleep(wait_time)
                else:
                    return f"[Error: {str(e)[:50]}]"
        return ""
    
    def get_embedding(self, text: str) -> np.ndarray:
        """埋め込みベクトル取得"""
        try:
            response = self.azure_client.embeddings.create(
                input=text,
                model=self.embedding_deployment
            )
            return np.array(response.data[0].embedding)
        except Exception as e:
            console.print(f"[red]Embedding error: {e}[/red]")
            return np.array([])
    
    def calculate_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """コサイン類似度計算"""
        if vec1.size == 0 or vec2.size == 0:
            return 0.0
        
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return float(dot_product / (norm1 * norm2))
    
    def analyze_all_terms(self, input_file: str) -> List[Dict]:
        """全用語の類似度分析"""
        # ファイル読み込み
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        all_terms = []
        for result in data.get("results", []):
            for term in result.get("terms", []):
                all_terms.append(term)
        
        console.print(f"[cyan]Analyzing {len(all_terms)} terms...[/cyan]")
        
        results = []
        for i, term_data in enumerate(track(all_terms, description="Processing")):
            term_name = term_data.get("term", "")
            original_def = term_data.get("definition", "")
            
            # レート制限対策
            if i > 0 and i % 3 == 0:
                time.sleep(3)
            
            # コンテキストなし定義生成（分野情報も渡す）
            field = term_data.get("metadata", {}).get("field", None)
            context_free_def = self.generate_definition(term_name, field)
            
            # 「定義不明」判定を追加
            is_unknown = False
            if context_free_def and any(keyword in context_free_def for keyword in 
                                       ["定義不明", "わかりません", "不明", "特定できません"]):
                is_unknown = True
                similarity = 0.0  # 明確に未知語として扱う
            elif context_free_def and not context_free_def.startswith("[Error"):
                # 埋め込み取得
                orig_embed = self.get_embedding(original_def)
                free_embed = self.get_embedding(context_free_def)
                
                # 類似度計算
                similarity = self.calculate_similarity(orig_embed, free_embed)
            else:
                similarity = -1  # エラーの場合
            
            results.append({
                "term": term_name,
                "original_definition": original_def[:100],
                "context_free_definition": context_free_def[:100] if context_free_def else "N/A",
                "similarity": similarity,
                "is_unknown": is_unknown,  # 追加
                "difficulty": term_data.get("metadata", {}).get("difficulty", 0),
                "field": term_data.get("metadata", {}).get("field", "N/A")
            })
        
        return results
    
    def display_results(self, results: List[Dict]):
        """結果表示"""
        # 類似度でソート（is_unknownフラグを優先）
        sorted_results = sorted(results, key=lambda x: (
            -1 if x.get("is_unknown") else x["similarity"] if x["similarity"] >= 0 else 999
        ))
        
        # 統計情報
        unknown_count = sum(1 for r in results if r.get("is_unknown"))
        valid_similarities = [r["similarity"] for r in results if r["similarity"] >= 0 and not r.get("is_unknown")]
        
        console.print("\n[bold green]== Similarity Statistics ==[/bold green]")
        console.print(f"Explicitly Unknown (定義不明): {unknown_count} terms")
        
        if valid_similarities:
            console.print(f"Mean: {np.mean(valid_similarities):.3f}")
            console.print(f"Median: {np.median(valid_similarities):.3f}")
            console.print(f"Std Dev: {np.std(valid_similarities):.3f}")
            console.print(f"Min: {np.min(valid_similarities):.3f}")
            console.print(f"Max: {np.max(valid_similarities):.3f}")
        
        # 閾値別カウント
        console.print("\n[bold cyan]== Threshold Analysis ==[/bold cyan]")
        thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        for threshold in thresholds:
            count = sum(1 for r in results if (0 <= r["similarity"] < threshold or r.get("is_unknown")))
            console.print(f"Below {threshold:.1f}: {count} terms (including {unknown_count} explicitly unknown)")
        
        # 全用語テーブル
        console.print("\n[bold magenta]== All Terms Sorted by Similarity ==[/bold magenta]")
        table = Table(show_header=True, header_style="bold")
        table.add_column("Term", style="cyan", width=25)
        table.add_column("Similarity", justify="right", width=10)
        table.add_column("Difficulty", justify="right", width=10)
        table.add_column("Field", width=15)
        table.add_column("Original Def", width=30)
        table.add_column("Context-Free Def", width=30)
        
        for result in sorted_results:
            sim_str = f"{result['similarity']:.3f}" if result['similarity'] >= 0 else "ERROR"
            
            # 色分け
            if result.get('is_unknown'):
                style = "bold magenta"  # 明確な未知語は紫
            elif result['similarity'] < 0:
                style = "red"
            elif result['similarity'] < 0.4:
                style = "bold red"
            elif result['similarity'] < 0.6:
                style = "yellow"
            elif result['similarity'] < 0.8:
                style = "green"
            else:
                style = "bold green"
            
            # is_unknownフラグの表示
            unknown_mark = " [U]" if result.get('is_unknown') else ""
            
            table.add_row(
                result["term"][:25] + unknown_mark,
                f"[{style}]{sim_str}[/{style}]",
                str(result["difficulty"]),
                result["field"][:15],
                result["original_definition"][:30] + "...",
                result["context_free_definition"][:30] + "..."
            )
        
        console.print(table)
    
    def save_results(self, results: List[Dict], output_file: str):
        """結果保存"""
        output_data = {
            "analysis_date": datetime.now().isoformat(),
            "total_terms": len(results),
            "statistics": {
                "valid_count": sum(1 for r in results if r["similarity"] >= 0),
                "error_count": sum(1 for r in results if r["similarity"] < 0),
                "explicitly_unknown_count": sum(1 for r in results if r.get("is_unknown"))
            },
            "results": sorted(results, key=lambda x: (
                -1 if x.get("is_unknown") else x["similarity"] if x["similarity"] >= 0 else 999
            ))
        }
        
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        console.print(f"\n[green]Results saved to {output_file}[/green]")


def main():
    """メイン関数"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Analyze similarities for all extracted terms"
    )
    parser.add_argument(
        "input",
        type=str,
        default="output/llm_terms.json",
        nargs="?",
        help="Input JSON file"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default="output/similarity_analysis.json",
        help="Output file"
    )
    parser.add_argument(
        "--llm-model",
        type=str,
        default="gemini-1.5-flash",
        help="LLM model for definitions"
    )
    
    args = parser.parse_args()
    
    # 分析実行
    analyzer = SimilarityAnalyzer(llm_model=args.llm_model)
    results = analyzer.analyze_all_terms(args.input)
    
    # 結果表示と保存
    analyzer.display_results(results)
    analyzer.save_results(results, args.output)


if __name__ == "__main__":
    main()