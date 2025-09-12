"""
専門用語抽出器の基底クラス
共通インターフェースと基本機能を提供
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import json

from pydantic import BaseModel, Field
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

from .document_loader import DocumentLoader

console = Console()


class Term(BaseModel):
    """専門用語のデータモデル"""
    term: str = Field(description="専門用語")
    definition: str = Field(description="用語の定義")
    score: float = Field(default=0.0, description="重要度スコア")
    frequency: int = Field(default=1, description="出現頻度")
    contexts: List[str] = Field(default_factory=list, description="出現文脈")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="追加メタデータ")


class ExtractionResult(BaseModel):
    """抽出結果のデータモデル"""
    file_path: str
    terms: List[Term]
    metadata: Dict[str, Any] = Field(default_factory=dict)
    processing_time: float = Field(default=0.0)


class BaseTermExtractor(ABC):
    """専門用語抽出器の基底クラス"""
    
    def __init__(self, chunk_size: int = 2000, chunk_overlap: int = 200):
        """
        初期化
        
        Args:
            chunk_size: テキスト分割時のチャンクサイズ
            chunk_overlap: チャンク間のオーバーラップ
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.document_loader = DocumentLoader()
    
    @abstractmethod
    async def extract_terms_from_text(self, text: str, metadata: Optional[Dict] = None) -> List[Term]:
        """
        テキストから専門用語を抽出（各実装クラスで定義）
        
        Args:
            text: 抽出対象のテキスト
            metadata: 追加のメタデータ
            
        Returns:
            抽出された専門用語のリスト
        """
        pass
    
    async def extract_from_file(self, file_path: Path) -> ExtractionResult:
        """
        ファイルから専門用語を抽出
        
        Args:
            file_path: 処理するファイルパス
            
        Returns:
            抽出結果
        """
        import time
        start_time = time.time()
        
        console.print(f"[cyan]処理中: {file_path.name}[/cyan]")
        
        # ファイルを読み込み
        text = self.document_loader.load(file_path)
        if not text:
            return ExtractionResult(
                file_path=str(file_path),
                terms=[],
                metadata={"error": "Failed to load file"}
            )
        
        # メタデータを取得
        metadata = self.document_loader.get_metadata(file_path)
        
        # 専門用語を抽出
        terms = await self.extract_terms_from_text(text, metadata)
        
        processing_time = time.time() - start_time
        
        return ExtractionResult(
            file_path=str(file_path),
            terms=terms,
            metadata=metadata,
            processing_time=processing_time
        )
    
    async def extract_from_directory(
        self, 
        directory: Path, 
        extensions: Optional[List[str]] = None
    ) -> List[ExtractionResult]:
        """
        ディレクトリ内の全ファイルから専門用語を抽出
        
        Args:
            directory: 処理するディレクトリ
            extensions: 処理する拡張子のリスト
            
        Returns:
            抽出結果のリスト
        """
        # ファイルを読み込み
        documents = self.document_loader.load_directory(directory, extensions)
        
        if not documents:
            console.print("[yellow]処理可能なファイルがありません[/yellow]")
            return []
        
        results = []
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task(
                f"[cyan]ファイル処理中... (合計: {len(documents)})",
                total=len(documents)
            )
            
            for file_path, text in documents.items():
                # メタデータを取得
                metadata = self.document_loader.get_metadata(file_path)
                
                # 専門用語を抽出
                terms = await self.extract_terms_from_text(text, metadata)
                
                # 結果を保存
                result = ExtractionResult(
                    file_path=str(file_path),
                    terms=terms,
                    metadata=metadata
                )
                results.append(result)
                
                progress.update(task, advance=1)
        
        return results
    
    def merge_terms(self, terms_list: List[List[Term]]) -> List[Term]:
        """
        複数のterm listを統合して重複を除去
        
        Args:
            terms_list: 統合するtermリストのリスト
            
        Returns:
            統合されたtermリスト
        """
        term_dict = {}
        
        for terms in terms_list:
            for term in terms:
                if term.term in term_dict:
                    # 既存の用語がある場合、情報を統合
                    existing = term_dict[term.term]
                    existing.frequency += term.frequency
                    existing.contexts.extend(term.contexts)
                    existing.score = max(existing.score, term.score)
                    
                    # より長い定義を採用
                    if len(term.definition) > len(existing.definition):
                        existing.definition = term.definition
                else:
                    term_dict[term.term] = term
        
        # スコアでソート
        return sorted(term_dict.values(), key=lambda x: x.score, reverse=True)
    
    def display_terms(self, terms: List[Term], limit: int = 20, title: str = "抽出された専門用語"):
        """
        専門用語を表形式で表示
        
        Args:
            terms: 表示する専門用語リスト
            limit: 表示する最大数
            title: テーブルのタイトル
        """
        if not terms:
            console.print("[yellow]専門用語が見つかりませんでした[/yellow]")
            return
        
        table = Table(title=f"{title} (上位{min(limit, len(terms))}件)")
        table.add_column("順位", justify="center", style="cyan")
        table.add_column("用語", style="green", no_wrap=True)
        table.add_column("スコア", justify="center")
        table.add_column("頻度", justify="center")
        table.add_column("定義", style="white", overflow="fold")
        
        for i, term in enumerate(terms[:limit], 1):
            # スコアによる色分け
            score_color = "red" if term.score >= 0.7 else "yellow" if term.score >= 0.4 else "white"
            
            # 定義を短縮
            definition = term.definition
            if len(definition) > 80:
                definition = definition[:77] + "..."
            
            table.add_row(
                str(i),
                term.term,
                f"[{score_color}]{term.score:.2f}[/{score_color}]",
                str(term.frequency),
                definition
            )
        
        console.print(table)
    
    def save_results(
        self, 
        results: List[ExtractionResult], 
        output_path: Path,
        format: str = "json"
    ):
        """
        結果をファイルに保存
        
        Args:
            results: 保存する結果
            output_path: 出力ファイルパス
            format: 出力形式（json, csv）
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format == "json":
            self._save_as_json(results, output_path)
        elif format == "csv":
            self._save_as_csv(results, output_path)
        else:
            console.print(f"[red]エラー: 未対応の出力形式: {format}[/red]")
    
    def _save_as_json(self, results: List[ExtractionResult], output_path: Path):
        """JSON形式で保存"""
        output_data = {
            "metadata": {
                "extraction_date": datetime.now().isoformat(),
                "extractor": self.__class__.__name__,
                "total_files": len(results),
                "total_terms": sum(len(r.terms) for r in results)
            },
            "results": []
        }
        
        for result in results:
            result_data = {
                "file_path": result.file_path,
                "processing_time": result.processing_time,
                "metadata": result.metadata,
                "terms": [
                    {
                        "term": term.term,
                        "definition": term.definition,
                        "score": term.score,
                        "frequency": term.frequency,
                        "contexts": term.contexts[:3],  # 最初の3つの文脈のみ
                        "metadata": term.metadata
                    }
                    for term in result.terms
                ]
            }
            output_data["results"].append(result_data)
        
        output_path.write_text(
            json.dumps(output_data, ensure_ascii=False, indent=2),
            encoding='utf-8'
        )
        
        console.print(f"[green]✓ 結果を保存しました: {output_path}[/green]")
    
    def _save_as_csv(self, results: List[ExtractionResult], output_path: Path):
        """CSV形式で保存"""
        import csv
        
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # ヘッダー
            writer.writerow(['ファイル', '用語', 'スコア', '頻度', '定義'])
            
            # データ
            for result in results:
                for term in result.terms:
                    writer.writerow([
                        Path(result.file_path).name,
                        term.term,
                        f"{term.score:.3f}",
                        term.frequency,
                        term.definition
                    ])
        
        console.print(f"[green]✓ 結果を保存しました: {output_path}[/green]")
    
    def get_statistics(self, results: List[ExtractionResult]) -> Dict[str, Any]:
        """
        抽出結果の統計情報を取得
        
        Args:
            results: 抽出結果リスト
            
        Returns:
            統計情報の辞書
        """
        all_terms = []
        for result in results:
            all_terms.extend(result.terms)
        
        if not all_terms:
            return {}
        
        scores = [term.score for term in all_terms]
        frequencies = [term.frequency for term in all_terms]
        
        return {
            "total_files": len(results),
            "total_terms": len(all_terms),
            "unique_terms": len(set(term.term for term in all_terms)),
            "avg_score": sum(scores) / len(scores),
            "max_score": max(scores),
            "min_score": min(scores),
            "avg_frequency": sum(frequencies) / len(frequencies),
            "max_frequency": max(frequencies),
            "total_processing_time": sum(r.processing_time for r in results)
        }