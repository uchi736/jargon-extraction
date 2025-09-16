#!/usr/bin/env python3
"""
LLMのみを使用した専門用語抽出システム（改良版）
2段階処理: 候補抽出 → 個別定義生成
"""

import sys
from pathlib import Path
import asyncio
from typing import List, Dict, Optional, Any, Tuple
import re

# プロジェクトルートをパスに追加
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.base_extractor import BaseTermExtractor, Term
from src.utils.document_loader import DocumentLoader

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from rich.console import Console
from rich.progress import track

# LangChain imports
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

load_dotenv()
console = Console()


class CandidateTerm(BaseModel):
    """専門用語候補モデル"""
    term: str = Field(description="専門用語")
    importance: int = Field(description="重要度（1-10）", ge=1, le=10)


class CandidateExtractionResult(BaseModel):
    """候補抽出結果"""
    candidates: List[CandidateTerm] = Field(description="専門用語候補リスト")


class TermDefinition(BaseModel):
    """用語定義モデル"""
    definition: str = Field(description="用語の詳細な定義（30-50語程度）")
    field: str = Field(description="分野")
    difficulty: int = Field(description="難易度（1-10）", ge=1, le=10)
    reasoning: str = Field(description="専門用語として重要な理由（50文字以内）")


class LLMTermExtractorV2(BaseTermExtractor):
    """改良版LLM専門用語抽出器（2段階処理）"""
    
    def __init__(self, deployment_name: str = "gpt-4.1-mini", max_terms: int = 30):
        """
        初期化
        
        Args:
            deployment_name: Azure OpenAIデプロイメント名
            max_terms: 抽出する最大用語数
        """
        # リカーシブ分割で500文字チャンク
        super().__init__(chunk_size=500, chunk_overlap=50)
        
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
        
        # リカーシブテキストスプリッター
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=["\n\n", "\n", "。", "、", " ", ""]
        )
        
        # 候補抽出用プロンプト
        self.candidate_prompt = ChatPromptTemplate.from_messages([
            ("system", """あなたは専門用語候補を識別するエキスパートです。
与えられたテキストから専門用語の候補を抽出してください。

抽出基準：
- その分野特有の概念や技術を表す用語
- 一般的すぎる単語（「データ」「情報」「システム」など）は除外
- 複合語や専門的な略語を優先的に抽出
- 重要度を1-10で評価（10が最重要）

{format_instructions}"""),
            ("human", """以下のテキストから専門用語候補を抽出してください。
最大{max_candidates}個まで重要度の高い順に選んでください。

テキスト:
{text}""")
        ])
        
        # 定義生成用プロンプト
        self.definition_prompt = ChatPromptTemplate.from_messages([
            ("system", """あなたは専門用語の定義を作成する専門家です。
提供された複数のコンテキストから、用語の包括的で正確な定義を作成してください。

定義作成の指針：
- 30-50語程度の詳細な定義
- 用語の本質的な意味と機能を説明
- 具体的な用途や応用例を含める
- 関連する概念や技術との関係性を明記
- 必要に応じて動作原理や特徴を説明
- 複数のコンテキストから総合的に判断

{format_instructions}"""),
            ("human", """専門用語「{term}」について、以下のコンテキストから定義を作成してください。

コンテキスト:
{contexts}

この用語の詳細な定義、分野、難易度、重要性を説明してください。""")
        ])
        
        # 出力パーサー
        self.candidate_parser = JsonOutputParser(pydantic_object=CandidateExtractionResult)
        self.definition_parser = JsonOutputParser(pydantic_object=TermDefinition)
    
    async def extract_terms_from_text(
        self, 
        text: str, 
        metadata: Optional[Dict] = None
    ) -> List[Term]:
        """
        2段階処理で専門用語を抽出
        
        Args:
            text: 抽出対象のテキスト
            metadata: 追加メタデータ
            
        Returns:
            抽出された専門用語リスト
        """
        console.print("[cyan]改良版2段階処理で専門用語を抽出中...[/cyan]")
        
        # 第1段階: 候補抽出
        candidates = await self._extract_candidates(text)
        console.print(f"[green]候補用語数: {len(candidates)}[/green]")
        
        # 第2段階: 各候補に定義を付与
        terms = []
        for i, candidate in enumerate(track(candidates[:self.max_terms], 
                                           description="定義生成中")):
            # 該当用語のすべてのコンテキストを収集
            contexts = self._find_all_contexts(text, candidate.term)
            
            if contexts:
                # 複数コンテキストから定義を生成
                definition_data = await self._generate_definition_with_contexts(
                    candidate.term, 
                    contexts
                )
                
                if definition_data:
                    term = Term(
                        term=candidate.term,
                        definition=definition_data.definition,
                        score=definition_data.difficulty / 10.0,
                        frequency=len(contexts),
                        contexts=contexts[:3],  # 最大3つのコンテキストを保存
                        metadata={
                            "field": definition_data.field,
                            "reasoning": definition_data.reasoning,
                            "difficulty": definition_data.difficulty,
                            "importance": candidate.importance
                        }
                    )
                    terms.append(term)
        
        return terms
    
    async def _extract_candidates(self, text: str) -> List[CandidateTerm]:
        """
        第1段階: 専門用語候補を抽出
        
        Args:
            text: 対象テキスト
            
        Returns:
            候補用語リスト
        """
        console.print("  [yellow]第1段階: 候補抽出中...[/yellow]")
        
        # テキストをチャンクに分割
        chunks = self.text_splitter.split_text(text)
        console.print(f"    チャンク数: {len(chunks)}")
        
        all_candidates = {}
        
        # 各チャンクから候補を抽出
        for i, chunk in enumerate(chunks):
            if i % 10 == 0:  # 進捗表示
                console.print(f"    処理中: {i}/{len(chunks)} チャンク")
            
            try:
                # 候補抽出チェーン
                chain = (
                    {
                        "text": lambda _: chunk,
                        "max_candidates": lambda _: 20,  # チャンクごとに20候補
                        "format_instructions": lambda _: self.candidate_parser.get_format_instructions()
                    }
                    | self.candidate_prompt
                    | self.llm
                    | self.candidate_parser
                )
                
                result = await chain.ainvoke({})
                
                # 候補を統合（重要度の高い方を保持）
                if isinstance(result, dict) and 'candidates' in result:
                    for candidate_data in result['candidates']:
                        candidate = CandidateTerm(**candidate_data)
                        if candidate.term not in all_candidates:
                            all_candidates[candidate.term] = candidate
                        else:
                            # より高い重要度を保持
                            if candidate.importance > all_candidates[candidate.term].importance:
                                all_candidates[candidate.term] = candidate
                                
            except Exception as e:
                continue  # エラーは無視して続行
        
        # 重要度でソート
        sorted_candidates = sorted(all_candidates.values(), 
                                 key=lambda x: x.importance, 
                                 reverse=True)
        
        return sorted_candidates
    
    def _find_all_contexts(self, text: str, term: str, window: int = 150) -> List[str]:
        """
        用語のすべての出現箇所からコンテキストを収集
        
        Args:
            text: 全文テキスト
            term: 対象用語
            window: コンテキストウィンドウサイズ
            
        Returns:
            コンテキストリスト（最大5つ）
        """
        contexts = []
        
        # 用語の全出現位置を検索
        pattern = re.compile(re.escape(term), re.IGNORECASE)
        matches = list(pattern.finditer(text))
        
        for match in matches[:5]:  # 最大5つのコンテキスト
            start_pos = match.start()
            end_pos = match.end()
            
            # 前後のコンテキストを取得
            context_start = max(0, start_pos - window)
            context_end = min(len(text), end_pos + window)
            
            context = text[context_start:context_end]
            
            # 文の境界で調整
            # 文頭を探す
            for sep in ['。', '．', '\n']:
                sep_pos = context.find(sep)
                if 0 < sep_pos < window//2:
                    context = context[sep_pos + 1:]
                    break
            
            # 文末を探す
            for sep in ['。', '．', '\n']:
                sep_pos = context.rfind(sep)
                if len(context) - window//2 < sep_pos < len(context) - 1:
                    context = context[:sep_pos + 1]
                    break
            
            contexts.append(context.strip())
        
        # 重複を除去
        unique_contexts = []
        for ctx in contexts:
            if ctx not in unique_contexts:
                unique_contexts.append(ctx)
        
        return unique_contexts
    
    async def _generate_definition_with_contexts(
        self, 
        term: str, 
        contexts: List[str]
    ) -> Optional[TermDefinition]:
        """
        第2段階: 複数コンテキストから定義を生成
        
        Args:
            term: 用語
            contexts: コンテキストリスト
            
        Returns:
            定義データ
        """
        try:
            # コンテキストを整形
            formatted_contexts = "\n\n".join([
                f"コンテキスト{i+1}:\n{ctx}" 
                for i, ctx in enumerate(contexts)
            ])
            
            # 定義生成チェーン
            chain = (
                {
                    "term": lambda _: term,
                    "contexts": lambda _: formatted_contexts,
                    "format_instructions": lambda _: self.definition_parser.get_format_instructions()
                }
                | self.definition_prompt
                | self.llm
                | self.definition_parser
            )
            
            result = await chain.ainvoke({})
            
            if isinstance(result, dict):
                return TermDefinition(**result)
                
        except Exception as e:
            console.print(f"[red]定義生成エラー ({term}): {e}[/red]")
        
        return None


async def main():
    """メイン処理"""
    if len(sys.argv) < 2:
        console.print("[red]使用法: python llm_extractor_v2.py <input_path> [output.json][/red]")
        console.print("  input_path: 文書ファイルまたはディレクトリ")
        console.print("  output.json: 出力ファイル（省略時: output/llm_terms_v2.json）")
        sys.exit(1)
    
    input_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2] if len(sys.argv) > 2 else "output/llm_terms_v2.json")
    
    # 抽出器を初期化
    extractor = LLMTermExtractorV2(max_terms=30)
    
    # 処理実行
    if input_path.is_file():
        result = await extractor.extract_from_file(input_path)
        results = [result]
    else:
        results = await extractor.extract_from_directory(input_path)
    
    if results:
        # 結果を表示
        all_terms = []
        for result in results:
            all_terms.extend(result.terms)
        
        extractor.display_terms(all_terms[:30], limit=30, title="改良版LLM抽出専門用語")
        
        # 結果を保存
        extractor.save_results(results, output_path, format="json")
        
        # 統計情報
        stats = extractor.get_statistics(results)
        console.print(f"\n[green]統計情報:[/green]")
        console.print(f"  処理ファイル数: {stats.get('total_files', 0)}")
        console.print(f"  抽出用語数: {stats.get('total_terms', 0)}")
        console.print(f"  処理時間: {stats.get('total_processing_time', 0):.2f}秒")


if __name__ == "__main__":
    asyncio.run(main())