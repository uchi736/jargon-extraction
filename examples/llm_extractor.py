#!/usr/bin/env python3
"""
LLMのみを使用した専門用語抽出システム（リファクタリング版）
共通モジュールを使用したクリーンな実装
"""

import sys
from pathlib import Path
import asyncio
from typing import List, Dict, Optional, Any

# プロジェクトルートをパスに追加
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.base_extractor import BaseTermExtractor, Term
from src.utils.document_loader import DocumentLoader

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from rich.console import Console

# LangChain imports
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
import os

load_dotenv()
console = Console()


class LLMTerm(BaseModel):
    """LLM用の専門用語モデル"""
    term: str = Field(description="専門用語")
    definition: str = Field(description="用語の詳細な定義（30-50語程度）")
    difficulty: int = Field(description="難易度（1-10）", ge=1, le=10)
    field: str = Field(description="分野")
    reasoning: str = Field(description="専門用語として選んだ理由（50文字以内）")


class LLMTermExtractionResult(BaseModel):
    """LLM抽出結果"""
    terms: List[LLMTerm] = Field(description="抽出された専門用語リスト")


class LLMTermExtractor(BaseTermExtractor):
    """LLMのみを使用した専門用語抽出器"""
    
    def __init__(self, deployment_name: str = "gpt-4.1-mini", max_terms: int = 30):
        """
        初期化
        
        Args:
            deployment_name: Azure OpenAIデプロイメント名
            max_terms: 抽出する最大用語数
        """
        super().__init__(chunk_size=4000, chunk_overlap=400)
        
        self.max_terms = max_terms
        
        # Azure OpenAI LLM設定
        self.llm = AzureChatOpenAI(
            deployment_name=deployment_name,
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            temperature=0.1,
            max_tokens=4096
        )
        
        # 出力パーサー
        self.output_parser = JsonOutputParser(pydantic_object=LLMTermExtractionResult)
        
        # プロンプトテンプレート
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """あなたは専門用語抽出のエキスパートです。
与えられたテキストから専門用語を抽出し、以下の基準で評価してください：

1. 専門性: その分野特有の概念や技術を表す用語
2. 難易度: 一般人にとっての理解の難しさ（1-10、10が最難）
3. 重要性: 文書内での重要度と出現頻度

抽出基準：
- 一般的すぎる単語（「データ」「情報」「システム」など）は除外
- 複合語や専門的な略語を優先的に抽出
- 各用語に詳細で学術的な定義を付与（30-50語程度）
  定義には以下を含める：
  * 用語の本質的な意味と機能
  * 具体的な用途や応用例
  * 関連する概念や技術との関係性
  * 必要に応じて動作原理や特徴
- なぜそれが専門用語なのか理由を明確に説明（50文字以内）

{format_instructions}"""),
            ("human", """以下のテキストから専門用語を抽出してください。
重要度の高い順に最大{max_terms}個の専門用語を選んでください。

テキスト:
{text}""")
        ])
    
    async def extract_terms_from_text(
        self, 
        text: str, 
        metadata: Optional[Dict] = None
    ) -> List[Term]:
        """
        LLMで専門用語を抽出
        
        Args:
            text: 抽出対象のテキスト
            metadata: 追加メタデータ
            
        Returns:
            抽出された専門用語リスト
        """
        console.print("[cyan]LLMで専門用語を抽出中...[/cyan]")
        
        # テキストをチャンクに分割
        chunks = self.document_loader.split_text(text, self.chunk_size, self.chunk_overlap)
        console.print(f"  チャンク数: {len(chunks)}")
        
        all_llm_terms = []
        
        # 各チャンクから専門用語を抽出
        for i, chunk in enumerate(chunks):
            console.print(f"  チャンク {i+1}/{len(chunks)} を処理中...")
            
            try:
                # LLMチェーンを構築
                chain = (
                    {
                        "text": lambda _: chunk,
                        "max_terms": lambda _: self.max_terms,
                        "format_instructions": lambda _: self.output_parser.get_format_instructions()
                    }
                    | self.prompt
                    | self.llm
                    | self.output_parser
                )
                
                # 抽出実行
                result = await chain.ainvoke({})
                
                if isinstance(result, dict) and 'terms' in result:
                    all_llm_terms.extend(result['terms'])
                
            except Exception as e:
                console.print(f"[yellow]警告: チャンク {i+1} の処理でエラー: {e}[/yellow]")
                continue
        
        # LLMTermをTermに変換して統合
        return self._convert_and_merge_llm_terms(all_llm_terms, text)
    
    def _convert_and_merge_llm_terms(self, llm_terms: List[Dict], text: str) -> List[Term]:
        """
        LLM用語を標準Termに変換して統合
        
        Args:
            llm_terms: LLMが抽出した用語リスト
            text: 元のテキスト（文脈抽出用）
            
        Returns:
            統合されたTermリスト
        """
        term_dict = {}
        
        for llm_term_data in llm_terms:
            if isinstance(llm_term_data, dict):
                # LLMTermの作成
                try:
                    llm_term = LLMTerm(**llm_term_data)
                except Exception:
                    continue
                
                term_str = llm_term.term
                
                if term_str not in term_dict:
                    # 文脈を抽出
                    contexts = self._extract_contexts(text, term_str)
                    
                    # Termに変換
                    term = Term(
                        term=term_str,
                        definition=llm_term.definition,
                        score=llm_term.difficulty / 10.0,  # 難易度をスコアに変換
                        frequency=len(contexts),  # 文脈数を頻度として使用
                        contexts=contexts,
                        metadata={
                            "field": llm_term.field,
                            "reasoning": llm_term.reasoning,
                            "difficulty": llm_term.difficulty
                        }
                    )
                    term_dict[term_str] = term
                else:
                    # 既存の用語がある場合、より詳細な定義を保持
                    existing = term_dict[term_str]
                    if len(llm_term.definition) > len(existing.definition):
                        existing.definition = llm_term.definition
                    # スコアは最大値を取る
                    existing.score = max(existing.score, llm_term.difficulty / 10.0)
        
        # スコアでソート
        sorted_terms = sorted(term_dict.values(), key=lambda x: x.score, reverse=True)
        return sorted_terms[:self.max_terms]
    
    def _extract_contexts(self, text: str, term: str, window: int = 100) -> List[str]:
        """
        用語の出現文脈を抽出
        
        Args:
            text: 対象テキスト
            term: 用語
            window: 文脈ウィンドウサイズ
            
        Returns:
            文脈リスト（最大3つ）
        """
        contexts = []
        index = 0
        
        while index < len(text):
            index = text.find(term, index)
            if index == -1:
                break
            
            # 前後の文脈を取得
            start = max(0, index - window)
            end = min(len(text), index + len(term) + window)
            
            # 文の境界を探す
            context_text = text[start:end]
            
            # 文頭を探す
            for sep in ['。', '．', '\n']:
                sep_pos = context_text.find(sep)
                if 0 < sep_pos < window:
                    context_text = context_text[sep_pos + 1:]
                    break
            
            # 文末を探す
            for sep in ['。', '．', '\n']:
                sep_pos = context_text.rfind(sep)
                if len(context_text) - window < sep_pos < len(context_text) - 1:
                    context_text = context_text[:sep_pos + 1]
                    break
            
            contexts.append(context_text.strip())
            index += len(term)
            
            # 最大3つまで
            if len(contexts) >= 3:
                break
        
        return contexts


async def main():
    """メイン処理"""
    if len(sys.argv) < 2:
        console.print("[red]使用法: python llm_extractor.py <input_path> [output.json][/red]")
        console.print("  input_path: 文書ファイルまたはディレクトリ")
        console.print("  output.json: 出力ファイル（省略時: output/llm_terms.json）")
        sys.exit(1)
    
    input_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2] if len(sys.argv) > 2 else "output/llm_terms.json")
    
    # 抽出器を初期化
    extractor = LLMTermExtractor(max_terms=30)
    
    # 処理実行
    if input_path.is_file():
        result = await extractor.extract_from_file(input_path)
        results = [result]
    else:
        results = await extractor.extract_from_directory(input_path)
    
    if results:
        # 全体の用語を統合
        all_terms = []
        for result in results:
            all_terms.extend(result.terms)
        
        # 重複を除去して表示
        merged_terms = extractor.merge_terms([all_terms])
        extractor.display_terms(merged_terms, limit=30, title="LLM抽出専門用語")
        
        # 結果を保存
        extractor.save_results(results, output_path, format="json")
        
        # 統計情報を表示
        stats = extractor.get_statistics(results)
        console.print(f"\n[green]統計情報:[/green]")
        console.print(f"  処理ファイル数: {stats.get('total_files', 0)}")
        console.print(f"  抽出用語数: {stats.get('total_terms', 0)}")
        console.print(f"  ユニーク用語数: {stats.get('unique_terms', 0)}")
        console.print(f"  平均スコア: {stats.get('avg_score', 0):.3f}")
        console.print(f"  処理時間: {stats.get('total_processing_time', 0):.2f}秒")


if __name__ == "__main__":
    asyncio.run(main())