#!/usr/bin/env python3
"""
改良版LLM抽出器のテスト
定義の品質と長さを確認
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from examples.llm_extractor import LLMTermExtractor
from rich.console import Console
from rich.table import Table

console = Console()

# テスト用のサンプルテキスト
SAMPLE_TEXT = """
アンモニア燃料エンジンの開発において、マイクロパイロット方式は重要な技術である。
この方式では、少量のディーゼル燃料を噴射して着火を促進し、
主燃料であるアンモニアの燃焼を安定化させる。
カーボンニュートラルを実現するため、N2O排出量の削減も重要な課題となっている。
インクラスターの国際規則に従い、GHG削減目標を達成する必要がある。
"""

async def test_extraction():
    """抽出テスト"""
    console.print("[bold cyan]改良版LLM抽出器のテスト[/bold cyan]\n")
    
    # 抽出器を初期化
    extractor = LLMTermExtractor(max_terms=10)
    
    # 用語抽出
    console.print("[yellow]用語を抽出中...[/yellow]")
    terms = await extractor.extract_terms_from_text(SAMPLE_TEXT)
    
    # 結果を表示
    console.print(f"\n[green]抽出された用語数: {len(terms)}[/green]\n")
    
    # 詳細テーブル
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("用語", style="cyan", width=20)
    table.add_column("定義", width=60)
    table.add_column("語数", justify="right", width=5)
    table.add_column("難易度", justify="right", width=5)
    table.add_column("分野", width=10)
    
    for term in terms[:5]:  # 最初の5つを表示
        # 定義の語数を計算（簡易的に空白で分割）
        word_count = len(term.definition.split())
        
        table.add_row(
            term.term,
            term.definition[:100] + ("..." if len(term.definition) > 100 else ""),
            str(word_count),
            str(term.metadata.get("difficulty", 0)),
            term.metadata.get("field", "N/A")
        )
    
    console.print(table)
    
    # 定義の詳細を表示
    console.print("\n[bold yellow]定義の詳細（最初の3つ）:[/bold yellow]")
    for i, term in enumerate(terms[:3], 1):
        console.print(f"\n{i}. [cyan]{term.term}[/cyan]")
        console.print(f"   定義: {term.definition}")
        console.print(f"   語数: {len(term.definition.split())}語")
        console.print(f"   文字数: {len(term.definition)}文字")
        console.print(f"   理由: {term.metadata.get('reasoning', 'N/A')}")
    
    return terms

if __name__ == "__main__":
    asyncio.run(test_extraction())