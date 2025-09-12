"""
共通文書ローダーモジュール
各種文書形式（PDF、Word、Markdown、HTML、TXT）の読み込みを統一的に処理
"""

from pathlib import Path
from typing import List, Optional, Dict, Any
import pymupdf  # PyMuPDF
import docx
from bs4 import BeautifulSoup
import markdown
from rich.console import Console

console = Console()


class DocumentLoader:
    """各種文書形式を統一的に読み込むローダー"""
    
    SUPPORTED_EXTENSIONS = {
        '.pdf': 'PDF',
        '.docx': 'Word',
        '.doc': 'Word',
        '.txt': 'Text',
        '.md': 'Markdown',
        '.html': 'HTML',
        '.htm': 'HTML'
    }
    
    @classmethod
    def load(cls, file_path: Path) -> Optional[str]:
        """
        ファイルを読み込んでテキストを返す
        
        Args:
            file_path: 読み込むファイルのパス
            
        Returns:
            抽出されたテキスト、失敗時はNone
        """
        if not file_path.exists():
            console.print(f"[red]エラー: ファイルが存在しません: {file_path}[/red]")
            return None
        
        suffix = file_path.suffix.lower()
        
        if suffix not in cls.SUPPORTED_EXTENSIONS:
            console.print(f"[yellow]警告: 未対応のファイル形式: {suffix}[/yellow]")
            return None
        
        try:
            if suffix == '.pdf':
                return cls._load_pdf(file_path)
            elif suffix in ['.docx', '.doc']:
                return cls._load_word(file_path)
            elif suffix == '.txt':
                return cls._load_text(file_path)
            elif suffix == '.md':
                return cls._load_markdown(file_path)
            elif suffix in ['.html', '.htm']:
                return cls._load_html(file_path)
        except Exception as e:
            console.print(f"[red]エラー: ファイル読み込み失敗 {file_path}: {e}[/red]")
            return None
    
    @staticmethod
    def _load_pdf(file_path: Path) -> str:
        """PDFファイルを読み込む"""
        doc = pymupdf.open(str(file_path))
        text_parts = []
        
        for page_num, page in enumerate(doc, start=1):
            text = page.get_text()
            if text.strip():
                text_parts.append(text)
        
        doc.close()
        
        if not text_parts:
            raise ValueError("PDFからテキストを抽出できませんでした")
        
        return "\n\n".join(text_parts)
    
    @staticmethod
    def _load_word(file_path: Path) -> str:
        """Wordファイルを読み込む"""
        doc = docx.Document(str(file_path))
        paragraphs = [para.text for para in doc.paragraphs if para.text.strip()]
        
        if not paragraphs:
            raise ValueError("Word文書からテキストを抽出できませんでした")
        
        return "\n\n".join(paragraphs)
    
    @staticmethod
    def _load_text(file_path: Path) -> str:
        """テキストファイルを読み込む"""
        # エンコーディングを自動検出
        encodings = ['utf-8', 'shift-jis', 'euc-jp', 'iso-2022-jp']
        
        for encoding in encodings:
            try:
                return file_path.read_text(encoding=encoding)
            except UnicodeDecodeError:
                continue
        
        # 全て失敗した場合はエラー無視で読み込み
        return file_path.read_text(encoding='utf-8', errors='ignore')
    
    @staticmethod
    def _load_markdown(file_path: Path) -> str:
        """Markdownファイルを読み込んでプレーンテキストに変換"""
        content = DocumentLoader._load_text(file_path)
        
        # MarkdownをHTMLに変換してからテキスト抽出
        html = markdown.markdown(content, extensions=['tables', 'fenced_code'])
        soup = BeautifulSoup(html, 'html.parser')
        
        return soup.get_text()
    
    @staticmethod
    def _load_html(file_path: Path) -> str:
        """HTMLファイルを読み込んでプレーンテキストに変換"""
        content = DocumentLoader._load_text(file_path)
        soup = BeautifulSoup(content, 'html.parser')
        
        # スクリプトとスタイルタグを除去
        for element in soup(['script', 'style', 'meta', 'link']):
            element.decompose()
        
        # テキストを抽出
        text = soup.get_text()
        
        # 空行を整理
        lines = [line.strip() for line in text.splitlines()]
        return "\n".join(line for line in lines if line)
    
    @classmethod
    def load_directory(cls, directory: Path, extensions: Optional[List[str]] = None) -> Dict[Path, str]:
        """
        ディレクトリ内の全対応ファイルを読み込む
        
        Args:
            directory: 読み込むディレクトリ
            extensions: 読み込む拡張子のリスト（Noneの場合は全対応形式）
            
        Returns:
            {ファイルパス: テキスト内容} の辞書
        """
        if not directory.is_dir():
            console.print(f"[red]エラー: ディレクトリが存在しません: {directory}[/red]")
            return {}
        
        # 拡張子リストの設定
        if extensions is None:
            extensions = list(cls.SUPPORTED_EXTENSIONS.keys())
        
        # ファイルを検索
        files = []
        for ext in extensions:
            files.extend(directory.glob(f"*{ext}"))
            files.extend(directory.glob(f"*{ext.upper()}"))
        
        # 重複を除去
        files = list(set(files))
        
        if not files:
            console.print(f"[yellow]警告: {directory} に処理可能なファイルがありません[/yellow]")
            return {}
        
        console.print(f"[green]見つかったファイル: {len(files)} 個[/green]")
        
        # 各ファイルを読み込み
        results = {}
        for file_path in sorted(files):
            console.print(f"読み込み中: {file_path.name}")
            text = cls.load(file_path)
            if text:
                results[file_path] = text
        
        return results
    
    @staticmethod
    def split_text(text: str, chunk_size: int = 2000, overlap: int = 200) -> List[str]:
        """
        テキストをチャンクに分割
        
        Args:
            text: 分割するテキスト
            chunk_size: チャンクサイズ（文字数）
            overlap: チャンク間のオーバーラップ（文字数）
            
        Returns:
            チャンクのリスト
        """
        if not text:
            return []
        
        chunks = []
        start = 0
        text_length = len(text)
        
        while start < text_length:
            # チャンクの終了位置を計算
            end = start + chunk_size
            
            # 最後のチャンクでない場合、文の境界を探す
            if end < text_length:
                # 句読点や改行を探して、そこで区切る
                for delimiter in ['。', '．', '\n', '！', '？', '. ']:
                    delimiter_pos = text.rfind(delimiter, start + chunk_size // 2, end)
                    if delimiter_pos != -1:
                        end = delimiter_pos + len(delimiter)
                        break
            else:
                end = text_length
            
            # チャンクを追加
            chunks.append(text[start:end])
            
            # 次の開始位置を設定（オーバーラップを考慮）
            start = end - overlap if end < text_length else text_length
        
        return chunks
    
    @staticmethod
    def get_metadata(file_path: Path) -> Dict[str, Any]:
        """
        ファイルのメタデータを取得
        
        Args:
            file_path: ファイルパス
            
        Returns:
            メタデータの辞書
        """
        stat = file_path.stat()
        
        return {
            'file_name': file_path.name,
            'file_path': str(file_path.absolute()),
            'file_size': stat.st_size,
            'extension': file_path.suffix.lower(),
            'modified_time': stat.st_mtime,
            'file_type': DocumentLoader.SUPPORTED_EXTENSIONS.get(
                file_path.suffix.lower(), 
                'Unknown'
            )
        }