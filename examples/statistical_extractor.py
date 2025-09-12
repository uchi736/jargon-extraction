#!/usr/bin/env python3
"""
統計的手法を使用した専門用語抽出システム
TF-IDF、N-gram、形態素解析を組み合わせた抽出
"""

import sys
from pathlib import Path
import asyncio
from typing import List, Dict, Optional, Any
from collections import Counter
import math

# プロジェクトルートをパスに追加
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.base_extractor import BaseTermExtractor, Term
from src.utils.document_loader import DocumentLoader

from dotenv import load_dotenv
from sudachipy import tokenizer, dictionary
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from rich.console import Console

# LangChain imports (for LLM validation)
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()
console = Console()


class StatisticalTermExtractor(BaseTermExtractor):
    """統計的手法による専門用語抽出器"""
    
    def __init__(
        self, 
        use_llm_validation: bool = True,
        min_term_length: int = 2,
        max_term_length: int = 10,
        min_frequency: int = 2
    ):
        """
        初期化
        
        Args:
            use_llm_validation: LLMによる検証を行うか
            min_term_length: 最小用語長
            max_term_length: 最大用語長  
            min_frequency: 最小出現頻度
        """
        super().__init__()
        
        # Sudachi tokenizer
        self.tokenizer = dictionary.Dictionary().create()
        
        # パラメータ
        self.min_term_length = min_term_length
        self.max_term_length = max_term_length
        self.min_frequency = min_frequency
        self.use_llm_validation = use_llm_validation
        
        # LLM設定（検証用）
        if use_llm_validation:
            self.llm = ChatGoogleGenerativeAI(
                model="gemini-2.0-flash-exp",
                temperature=0.1
            )
            
            self.validation_prompt = ChatPromptTemplate.from_messages([
                ("system", "あなたは専門用語判定のエキスパートです。"),
                ("human", """以下の候補が専門用語かどうか判定してください。
                
候補語: {term}
文脈: {context}

専門用語の場合は「YES」、そうでない場合は「NO」と答え、
その後に簡潔な定義（30文字以内）を付けてください。

形式: YES/NO | 定義""")
            ])
    
    async def extract_terms_from_text(
        self, 
        text: str, 
        metadata: Optional[Dict] = None
    ) -> List[Term]:
        """
        統計的手法で専門用語を抽出
        
        Args:
            text: 抽出対象のテキスト
            metadata: 追加メタデータ
            
        Returns:
            抽出された専門用語リスト
        """
        console.print("[cyan]統計的手法で専門用語を抽出中...[/cyan]")
        
        # 1. 形態素解析で候補を生成
        candidates = self._extract_candidates(text)
        console.print(f"  候補語数: {len(candidates)}")
        
        # 2. TF-IDFスコアを計算
        tfidf_scores = self._calculate_tfidf(text, list(candidates.keys()))
        
        # 3. 複合語スコア（C-value）を計算
        cvalue_scores = self._calculate_cvalue(candidates)
        
        # 4. 統合スコアを計算
        terms = []
        for term, frequency in candidates.items():
            if frequency < self.min_frequency:
                continue
            
            # スコア統合
            tfidf = tfidf_scores.get(term, 0.0)
            cvalue = cvalue_scores.get(term, 0.0)
            
            # 重み付き平均
            score = (tfidf * 0.5 + cvalue * 0.5)
            
            # 文脈を抽出
            contexts = self._extract_contexts(text, term)
            
            terms.append(Term(
                term=term,
                definition="",  # 後でLLMで生成
                score=score,
                frequency=frequency,
                contexts=contexts,
                metadata={"tfidf": tfidf, "cvalue": cvalue}
            ))
        
        # スコアでソート
        terms.sort(key=lambda x: x.score, reverse=True)
        
        # 5. LLM検証（オプション）
        if self.use_llm_validation and terms:
            terms = await self._validate_with_llm(terms[:50], text)  # 上位50件を検証
        
        return terms[:30]  # 上位30件を返す
    
    def _extract_candidates(self, text: str) -> Dict[str, int]:
        """
        形態素解析で専門用語候補を抽出
        
        Args:
            text: 対象テキスト
            
        Returns:
            {候補語: 出現頻度} の辞書
        """
        candidates = Counter()
        
        # Mode.Aで形態素解析（短単位）
        tokens = self.tokenizer.tokenize(text, tokenizer.Tokenizer.SplitMode.A)
        
        # 単一名詞と複合名詞を抽出
        i = 0
        while i < len(tokens):
            token = tokens[i]
            pos = token.part_of_speech()[0]
            
            # 名詞の場合
            if pos == "名詞":
                # 複合名詞を構築
                compound = [token.surface()]
                j = i + 1
                
                while j < len(tokens):
                    next_token = tokens[j]
                    next_pos = next_token.part_of_speech()[0]
                    
                    # 連続する名詞を結合
                    if next_pos == "名詞":
                        compound.append(next_token.surface())
                        j += 1
                    else:
                        break
                
                # 複合語を候補に追加
                for length in range(1, min(len(compound) + 1, self.max_term_length + 1)):
                    for start in range(len(compound) - length + 1):
                        term = "".join(compound[start:start + length])
                        
                        # 長さチェック
                        if self.min_term_length <= len(term) <= self.max_term_length * 2:
                            candidates[term] += 1
                
                i = j
            else:
                i += 1
        
        # カタカナ語も抽出
        import re
        katakana_pattern = re.compile(r'[ァ-ヴー]+')
        for match in katakana_pattern.finditer(text):
            term = match.group()
            if self.min_term_length <= len(term):
                candidates[term] += 1
        
        # 英数字混在パターンも抽出
        alphanumeric_pattern = re.compile(r'[A-Za-z][A-Za-z0-9\-]*[A-Za-z0-9]')
        for match in alphanumeric_pattern.finditer(text):
            term = match.group()
            if self.min_term_length <= len(term):
                candidates[term] += 1
        
        return dict(candidates)
    
    def _calculate_tfidf(self, text: str, terms: List[str]) -> Dict[str, float]:
        """
        TF-IDFスコアを計算
        
        Args:
            text: 対象テキスト
            terms: 評価する用語リスト
            
        Returns:
            {用語: TF-IDFスコア} の辞書
        """
        # 文書を文単位に分割
        sentences = text.split('。')
        sentences = [s for s in sentences if s.strip()]
        
        if not sentences:
            return {term: 0.0 for term in terms}
        
        # TF-IDF計算
        try:
            vectorizer = TfidfVectorizer(
                vocabulary={term: i for i, term in enumerate(terms)},
                token_pattern=r'\S+'
            )
            
            # 各文をドキュメントとして扱う
            tfidf_matrix = vectorizer.fit_transform(sentences)
            
            # 各用語の最大TF-IDFスコアを取得
            scores = {}
            for i, term in enumerate(terms):
                column = tfidf_matrix[:, i].toarray().flatten()
                scores[term] = float(np.max(column)) if column.size > 0 else 0.0
            
            return scores
            
        except Exception as e:
            console.print(f"[yellow]TF-IDF計算エラー: {e}[/yellow]")
            return {term: 0.0 for term in terms}
    
    def _calculate_cvalue(self, candidates: Dict[str, int]) -> Dict[str, float]:
        """
        C-value（複合語重要度）を計算
        
        Args:
            candidates: {候補語: 頻度} の辞書
            
        Returns:
            {候補語: C-value} の辞書
        """
        cvalues = {}
        
        for term in candidates:
            frequency = candidates[term]
            length = len(term)
            
            # 部分文字列として含まれる回数を計算
            contained_count = 0
            containing_terms = 0
            
            for other_term in candidates:
                if term != other_term and term in other_term:
                    contained_count += candidates[other_term]
                    containing_terms += 1
            
            # C-value計算
            if containing_terms > 0:
                cvalue = math.log2(length + 1) * (frequency - contained_count / containing_terms)
            else:
                cvalue = math.log2(length + 1) * frequency
            
            # 正規化（0-1の範囲に）
            cvalues[term] = max(0, cvalue / 100)  # 100で割って正規化
        
        return cvalues
    
    def _extract_contexts(self, text: str, term: str, window: int = 50) -> List[str]:
        """
        用語の出現文脈を抽出
        
        Args:
            text: 対象テキスト
            term: 用語
            window: 文脈ウィンドウサイズ
            
        Returns:
            文脈リスト
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
            context = text[start:end]
            
            # 文の境界で切る
            if start > 0:
                context = "..." + context
            if end < len(text):
                context = context + "..."
            
            contexts.append(context)
            index += len(term)
            
            # 最大3つまで
            if len(contexts) >= 3:
                break
        
        return contexts
    
    async def _validate_with_llm(self, terms: List[Term], text: str) -> List[Term]:
        """
        LLMで専門用語を検証
        
        Args:
            terms: 検証する用語リスト
            text: 元のテキスト
            
        Returns:
            検証済みの用語リスト
        """
        console.print("[cyan]LLMで専門用語を検証中...[/cyan]")
        
        validated_terms = []
        chain = self.validation_prompt | self.llm
        
        for term in terms:
            try:
                # 文脈を準備
                context = term.contexts[0] if term.contexts else text[:200]
                
                # LLMに問い合わせ
                response = await chain.ainvoke({
                    "term": term.term,
                    "context": context
                })
                
                # レスポンスを解析
                content = response.content.strip()
                if content.startswith("YES"):
                    # 定義を抽出
                    parts = content.split("|", 1)
                    if len(parts) > 1:
                        term.definition = parts[1].strip()
                    else:
                        term.definition = f"{term.term}に関する専門用語"
                    
                    validated_terms.append(term)
                    
            except Exception as e:
                console.print(f"[yellow]検証エラー ({term.term}): {e}[/yellow]")
                # エラーの場合は含める（安全側に倒す）
                term.definition = f"{term.term}（定義生成失敗）"
                validated_terms.append(term)
        
        return validated_terms


async def main():
    """メイン処理"""
    if len(sys.argv) < 2:
        console.print("[red]使用法: python statistical_extractor.py <input_path> [output.json][/red]")
        sys.exit(1)
    
    input_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2] if len(sys.argv) > 2 else "output/statistical_terms.json")
    
    # 抽出器を初期化
    extractor = StatisticalTermExtractor(
        use_llm_validation=True,
        min_frequency=2
    )
    
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
        
        merged_terms = extractor.merge_terms([all_terms])
        extractor.display_terms(merged_terms)
        
        # 保存
        extractor.save_results(results, output_path)
        
        # 統計を表示
        stats = extractor.get_statistics(results)
        console.print(f"\n[green]統計情報:[/green]")
        for key, value in stats.items():
            console.print(f"  {key}: {value}")


if __name__ == "__main__":
    asyncio.run(main())