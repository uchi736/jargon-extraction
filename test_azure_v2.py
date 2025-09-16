#!/usr/bin/env python3
"""
Azure OpenAI版のテスト（小規模）
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from examples.llm_extractor_v2 import LLMTermExtractorV2
from rich.console import Console

console = Console()

# 短いテストテキスト
SAMPLE_TEXT = """
マイクロパイロット方式は、アンモニア燃料エンジンの重要な技術である。
この方式により、アンモニアの低い着火性を補完できる。
"""

async def test_azure():
    """Azure OpenAI版のテスト"""
    console.print("[bold cyan]Azure OpenAI (gpt-4.1-mini) テスト[/bold cyan]\n")
    
    # 抽出器を初期化（少数の用語のみ）
    extractor = LLMTermExtractorV2(deployment_name="gpt-4.1-mini", max_terms=3)
    
    # 用語抽出
    console.print("[yellow]抽出中...[/yellow]")
    terms = await extractor.extract_terms_from_text(SAMPLE_TEXT)
    
    # 結果表示
    console.print(f"\n[green]抽出された用語数: {len(terms)}[/green]")
    
    for i, term in enumerate(terms, 1):
        console.print(f"\n{i}. [cyan]{term.term}[/cyan]")
        console.print(f"   定義: {term.definition[:100]}...")
        console.print(f"   分野: {term.metadata.get('field', 'N/A')}")
    
    console.print("\n[green]Azure OpenAI版が正常に動作しています！[/green]")

if __name__ == "__main__":
    asyncio.run(test_azure())