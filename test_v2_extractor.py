#!/usr/bin/env python3
"""
改良版2段階抽出器のテスト
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from examples.llm_extractor_v2 import LLMTermExtractorV2
from rich.console import Console
from rich.table import Table

console = Console()

# テスト用のサンプルテキスト
SAMPLE_TEXT = """
アンモニア燃料エンジンの開発において、マイクロパイロット方式は重要な技術である。
この方式では、少量のディーゼル燃料を噴射して着火を促進し、
主燃料であるアンモニアの燃焼を安定化させる。

マイクロパイロット方式の利点は、アンモニアの低い着火性を補完できることである。
アンモニアは自着火温度が高く、火炎伝播速度も遅いため、
通常の点火方式では安定した燃焼が困難である。

カーボンニュートラルを実現するため、N2O排出量の削減も重要な課題となっている。
N2Oは強力な温室効果ガスであり、その削減は必須である。

インクラスターの国際規則に従い、GHG削減目標を達成する必要がある。
マイクロパイロット方式を採用することで、効率的な燃焼が可能となる。
"""

async def test_v2_extraction():
    """2段階処理のテスト"""
    console.print("[bold cyan]改良版2段階抽出器のテスト[/bold cyan]\n")
    
    # 抽出器を初期化
    extractor = LLMTermExtractorV2(max_terms=5)
    
    # 用語抽出
    console.print("[yellow]2段階処理で用語を抽出中...[/yellow]")
    terms = await extractor.extract_terms_from_text(SAMPLE_TEXT)
    
    # 結果を表示
    console.print(f"\n[green]抽出された用語数: {len(terms)}[/green]\n")
    
    # 詳細表示
    for i, term in enumerate(terms, 1):
        console.print(f"\n[bold cyan]{i}. {term.term}[/bold cyan]")
        console.print(f"   [yellow]定義:[/yellow] {term.definition}")
        console.print(f"   [yellow]分野:[/yellow] {term.metadata.get('field', 'N/A')}")
        console.print(f"   [yellow]難易度:[/yellow] {term.metadata.get('difficulty', 0)}/10")
        console.print(f"   [yellow]重要度:[/yellow] {term.metadata.get('importance', 0)}/10")
        console.print(f"   [yellow]出現回数:[/yellow] {term.frequency}")
        console.print(f"   [yellow]理由:[/yellow] {term.metadata.get('reasoning', 'N/A')}")
        
        # コンテキスト表示
        if term.contexts:
            console.print(f"   [yellow]コンテキスト例:[/yellow]")
            for j, ctx in enumerate(term.contexts[:2], 1):
                console.print(f"     {j}. {ctx[:100]}...")
    
    # 比較テーブル
    console.print("\n[bold magenta]処理方式の比較:[/bold magenta]")
    table = Table(show_header=True, header_style="bold")
    table.add_column("項目", width=20)
    table.add_column("従来版", width=30)
    table.add_column("改良版（2段階）", width=30)
    
    table.add_row(
        "候補抽出",
        "チャンクごとに定義も同時生成",
        "候補のみ高速抽出"
    )
    table.add_row(
        "定義生成",
        "チャンク内の文脈のみ",
        "全文から複数コンテキスト収集"
    )
    table.add_row(
        "チャンクサイズ",
        "4000文字",
        "500文字（リカーシブ）"
    )
    table.add_row(
        "API呼び出し",
        "チャンク数回",
        "チャンク数 + 用語数回"
    )
    table.add_row(
        "定義の質",
        "局所的な文脈",
        "包括的な理解"
    )
    
    console.print(table)
    
    return terms

if __name__ == "__main__":
    asyncio.run(test_v2_extraction())