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
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()
console = Console()


class StatisticalTermExtractor(BaseTermExtractor):
    """統計的手法による専門用語抽出器"""

    # 日本語ストップワードリスト
    JAPANESE_STOPWORDS = {
        'こと', 'もの', 'とき', 'ところ', 'ため', 'よう', 'ほう',
        'わけ', 'はず', 'つもり', 'とおり', 'まま', 'たち', 'さん',
        'みたい', 'そう', 'ごと', 'なか', 'うえ', 'した', 'あと',
        'まえ', 'つぎ', 'ほか', 'なに', 'どこ', 'だれ', 'いつ',
        'どう', 'どれ', 'これ', 'それ', 'あれ', 'ここ', 'そこ',
        'あそこ', 'こちら', 'そちら', 'あちら', 'どちら', 'など',
        'ら', 'たり', 'だけ', 'ばかり', 'ぐらい', 'くらい', 'ほど'
    }

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
            import os
            self.llm = AzureChatOpenAI(
                api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                azure_deployment="gpt-4.1-mini",
                temperature=0.1
            )
            
            self.validation_prompt = ChatPromptTemplate.from_messages([
                ("system", """あなたは専門用語候補を識別するエキスパートです。
与えられた用語が、その分野特有の専門用語かどうかを判定してください。

専門用語の判定基準：
- その分野特有の概念、技術、製品、手法を表す用語
- 一般的な辞書には載っていない、または特殊な意味で使われている用語
- 業界や学術分野で共通理解される専門的な概念

除外基準（一般的すぎる用語）：
- 日常会話でも使われる一般名詞（「データ」「情報」「システム」「管理」「処理」など）
- 抽象的で文脈に依存しない概念（「効率」「改善」「評価」「分析」など）
- 単純な数値や単位（ただし特殊な単位は専門用語とする）

判定時の注意：
- 文脈での使われ方を重視
- 複合語は構成要素が一般的でも、組み合わせが専門的なら専門用語とする
- 略語やアルファベット表記は多くの場合専門用語"""),
                ("human", """以下の用語について判定してください。

用語: {term}
文脈: {context}

判定結果：
- 専門用語の場合: YES | その分野と簡潔な定義（30文字以内）
- 一般用語の場合: NO | 理由（例：一般的な概念）

形式: YES/NO | 説明""")
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
        console.print(f"  [yellow]形態素解析による初期候補数: {len(candidates)}[/yellow]")

        # デバッグ: 上位候補を表示
        if len(candidates) > 0:
            sorted_candidates = sorted(candidates.items(), key=lambda x: x[1], reverse=True)[:10]
            console.print("  [dim]頻度上位10件:[/dim]")
            for term, freq in sorted_candidates:
                console.print(f"    - {term}: {freq}回")
        
        # 2. TF-IDFスコアを計算
        tfidf_scores = self._calculate_tfidf(text, list(candidates.keys()))
        
        # 3. 複合語スコア（C-value）を計算
        cvalue_scores = self._calculate_cvalue(candidates)
        
        # 4. 統合スコアを計算
        terms = []
        filtered_count = 0
        for term, frequency in candidates.items():
            if frequency < self.min_frequency:
                filtered_count += 1
                continue
            
            # スコア統合
            tfidf = tfidf_scores.get(term, 0.0)
            cvalue = cvalue_scores.get(term, 0.0)
            
            # 重み付き平均（C値を強く重視）
            score = (tfidf * 0.2 + cvalue * 0.8)  # C値をさらに重視
            
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

        # 包含関係によるペナルティを適用
        terms = self._apply_substring_penalty(terms)

        # 再度ソート
        terms.sort(key=lambda x: x.score, reverse=True)

        console.print(f"  [yellow]最小頻度({self.min_frequency})でフィルタ: {filtered_count}件除外[/yellow]")
        console.print(f"  [yellow]スコア計算後の候補数: {len(terms)}[/yellow]")

        # デバッグ: スコア上位を表示
        if len(terms) > 0:
            console.print("  [dim]スコア上位10件 (TF-IDF + C-value):[/dim]")
            for i, term in enumerate(terms[:10]):
                tfidf = term.metadata.get('tfidf', 0)
                cvalue = term.metadata.get('cvalue', 0)
                console.print(f"    {i+1}. {term.term}: スコア={term.score:.3f} (TF-IDF={tfidf:.3f}, C-val={cvalue:.3f}, 頻度={term.frequency})")
        
        # 5. LLM検証（オプション）
        if self.use_llm_validation and terms:
            console.print(f"  [cyan]LLM検証対象: 上位{min(50, len(terms))}件[/cyan]")
            terms = await self._validate_with_llm(terms[:50], text)  # 上位50件を検証
            console.print(f"  [yellow]LLM検証後の候補数: {len(terms)}[/yellow]")

        final_terms = terms[:50]  # 上位50件を返す
        console.print(f"  [green]最終的な専門用語数: {len(final_terms)}[/green]")
        return final_terms
    
    def _extract_candidates(self, text: str) -> Dict[str, int]:
        """
        形態素解析で専門用語候補を抽出

        Args:
            text: 対象テキスト

        Returns:
            {候補語: 出現頻度} の辞書
        """
        candidates = Counter()

        # Sudachiの入力制限（49149バイト）を考慮してテキストを分割
        max_bytes = 40000  # 安全マージンを持たせる
        text_chunks = []

        if len(text.encode('utf-8')) > max_bytes:
            # テキストを文単位で分割
            sentences = text.split('。')
            current_chunk = ""

            for sentence in sentences:
                if len((current_chunk + sentence + '。').encode('utf-8')) < max_bytes:
                    current_chunk += sentence + '。'
                else:
                    if current_chunk:
                        text_chunks.append(current_chunk)
                    current_chunk = sentence + '。'

            if current_chunk:
                text_chunks.append(current_chunk)
        else:
            text_chunks = [text]

        # 各チャンクを処理
        for chunk in text_chunks:
            if not chunk.strip():
                continue

            # Mode.Aで形態素解析（短単位）
            tokens = self.tokenizer.tokenize(chunk, tokenizer.Tokenizer.SplitMode.A)

            # 単一名詞と複合名詞を抽出
            i = 0
            while i < len(tokens):
                token = tokens[i]
                pos_features = token.part_of_speech()
                pos = pos_features[0]

                # 名詞の場合
                if pos == "名詞":
                    # 品詞細分類をチェック（非自立、代名詞、数詞を除外）
                    if len(pos_features) > 1:
                        subpos = pos_features[1]
                        if subpos in ["非自立", "代名詞", "数"]:
                            i += 1
                            continue

                    # ストップワードチェック
                    if token.surface() in self.JAPANESE_STOPWORDS:
                        i += 1
                        continue

                    # 複合名詞を構築
                    compound = []
                    compound_tokens = []

                    # 現在のトークンを追加（ストップワードでなければ）
                    if token.surface() not in self.JAPANESE_STOPWORDS:
                        compound.append(token.surface())
                        compound_tokens.append(token)

                    j = i + 1

                    while j < len(tokens):
                        next_token = tokens[j]
                        next_pos_features = next_token.part_of_speech()
                        next_pos = next_pos_features[0]

                        # 連続する名詞を結合
                        if next_pos == "名詞":
                            # 品詞細分類チェック
                            if len(next_pos_features) > 1:
                                next_subpos = next_pos_features[1]
                                if next_subpos in ["非自立", "代名詞", "数"]:
                                    break

                            # ストップワードが末尾に来た場合は終了
                            if next_token.surface() in self.JAPANESE_STOPWORDS:
                                break

                            compound.append(next_token.surface())
                            compound_tokens.append(next_token)
                            j += 1
                        else:
                            break

                    # 複合語を候補に追加
                    if compound:  # 空でない場合のみ処理
                        for length in range(1, min(len(compound) + 1, self.max_term_length + 1)):
                            for start in range(len(compound) - length + 1):
                                term = "".join(compound[start:start + length])

                                # ストップワードで始まる・終わる語は除外
                                if self._is_valid_term(term):
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
            if self.min_term_length <= len(term) and self._is_valid_term(term):
                candidates[term] += 1

        # 英数字混在パターンも抽出
        alphanumeric_pattern = re.compile(r'[A-Za-z][A-Za-z0-9\-]*[A-Za-z0-9]')
        for match in alphanumeric_pattern.finditer(text):
            term = match.group()
            if self.min_term_length <= len(term):
                candidates[term] += 1

        # 1文字の候補や頻度が極端に低いものを除外
        filtered_candidates = {}
        for term, freq in candidates.items():
            if len(term) > 1:  # 1文字は除外
                filtered_candidates[term] = freq

        return filtered_candidates

    def _is_valid_term(self, term: str) -> bool:
        """
        用語が有効かチェック

        Args:
            term: チェックする用語

        Returns:
            有効な場合True
        """
        # ストップワードチェック
        if term in self.JAPANESE_STOPWORDS:
            return False

        # 末尾がストップワードで終わる複合語を除外
        for stopword in self.JAPANESE_STOPWORDS:
            if term.endswith(stopword) and len(term) > len(stopword):
                return False

        # 先頭がストップワードで始まる複合語を除外
        for stopword in self.JAPANESE_STOPWORDS:
            if term.startswith(stopword) and len(term) > len(stopword):
                return False

        return True
    
    def _calculate_tfidf(self, text: str, terms: List[str]) -> Dict[str, float]:
        """
        TF-IDFスコアを計算（適切なチャンク分割で）

        Args:
            text: 対象テキスト
            terms: 評価する用語リスト

        Returns:
            {用語: TF-IDFスコア} の辞書
        """
        # 文書を段落または適切なチャンクに分割
        chunks = self._split_into_chunks(text)

        if not chunks:
            return {term: 0.0 for term in terms}

        console.print(f"  [dim]TF-IDF計算: {len(chunks)}チャンクで処理[/dim]")

        # デバッグ: チャンクの内容を確認
        for i, chunk in enumerate(chunks[:3]):  # 最初の3チャンクを表示
            console.print(f"    [dim]チャンク{i+1}: {chunk[:50]}...[/dim]")

        # TF-IDF計算
        try:
            # シンプルなトークナイザー（語彙を事前指定しているので）
            def custom_tokenizer(text):
                # 空白で分割（TfidfVectorizerのvocabularyと一致させるため）
                # 各用語がテキスト内に出現するかをチェック
                result = []
                for term in terms:
                    if term in text:
                        result.append(term)
                return result if result else ['EMPTY']  # 空の場合のフォールバック

            # 各チャンクで用語の出現をカウント
            doc_term_matrix = []
            for chunk in chunks:
                doc_vector = []
                for term in terms:
                    # 単純な出現回数をカウント
                    count = chunk.count(term)
                    doc_vector.append(count)
                doc_term_matrix.append(doc_vector)

            # numpy配列に変換
            doc_term_matrix = np.array(doc_term_matrix)

            # 手動でTF-IDFを計算
            scores = {}
            num_docs = len(chunks)

            for i, term in enumerate(terms):
                # Term Frequency (TF)
                term_freqs = doc_term_matrix[:, i]

                # Document Frequency (DF)
                doc_freq = np.sum(term_freqs > 0)

                if doc_freq == 0:
                    scores[term] = 0.0
                    continue

                # Inverse Document Frequency (IDF)
                idf = math.log((num_docs + 1) / (doc_freq + 1)) + 1

                # TF-IDF
                tf_idf_scores = []
                for tf in term_freqs:
                    if tf > 0:
                        # 対数スケール化したTF
                        tf_log = 1 + math.log(tf)
                        tf_idf_scores.append(tf_log * idf)
                    else:
                        tf_idf_scores.append(0)

                # 最大値を取得
                max_score = max(tf_idf_scores) if tf_idf_scores else 0
                scores[term] = float(max_score)

            # スコアを正規化（0-1の範囲に）
            max_overall = max(scores.values()) if scores else 1
            if max_overall > 0:
                scores = {k: v / max_overall for k, v in scores.items()}

            # デバッグ: ゼロのTF-IDFを持つ用語を表示
            zero_terms = [term for term, score in scores.items() if score == 0]
            if zero_terms:
                console.print(f"    [dim]TF-IDF=0の用語: {', '.join(zero_terms[:10])}...[/dim]")

            # スコアが計算された

            return scores

        except Exception as e:
            console.print(f"[yellow]TF-IDF計算エラー: {e}[/yellow]")
            import traceback
            traceback.print_exc()
            return {term: 0.0 for term in terms}

    def _split_into_chunks(self, text: str, chunk_size: int = 500) -> List[str]:
        """
        テキストを適切なチャンクに分割

        Args:
            text: 分割対象のテキスト
            chunk_size: 目標チャンクサイズ（文字数）

        Returns:
            チャンクのリスト
        """
        chunks = []

        # まず段落で分割を試みる
        paragraphs = text.split('\n\n')

        for para in paragraphs:
            if not para.strip():
                continue

            # 段落が長すぎる場合は文で分割
            if len(para) > chunk_size * 2:
                sentences = para.split('。')
                current_chunk = ""

                for sent in sentences:
                    if not sent.strip():
                        continue

                    if len(current_chunk) + len(sent) < chunk_size:
                        current_chunk += sent + '。'
                    else:
                        if current_chunk:
                            chunks.append(current_chunk)
                        current_chunk = sent + '。'

                if current_chunk:
                    chunks.append(current_chunk)
            else:
                # 段落が適切なサイズならそのまま追加
                chunks.append(para)

        # チャンクが少なすぎる場合は文単位で分割
        if len(chunks) < 5:
            chunks = []
            sentences = text.split('。')
            current_chunk = ""
            sent_count = 0

            for sent in sentences:
                if not sent.strip():
                    continue

                current_chunk += sent + '。'
                sent_count += 1

                # 5文ごとまたは一定文字数でチャンク化
                if sent_count >= 5 or len(current_chunk) > chunk_size:
                    chunks.append(current_chunk)
                    current_chunk = ""
                    sent_count = 0

            if current_chunk:
                chunks.append(current_chunk)

        return chunks
    
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

    def _apply_substring_penalty(self, terms: List[Term]) -> List[Term]:
        """
        部分文字列関係にある用語にペナルティを適用

        Args:
            terms: 評価済みの用語リスト

        Returns:
            ペナルティ適用後の用語リスト
        """
        console.print("  [dim]包含関係ペナルティを適用中...[/dim]")

        # 用語をグループ化（同じベースを持つ用語群）
        term_groups = {}

        for term in terms:
            # 簡単なグルーピング：他の用語に含まれるかチェック
            is_substring = False
            containing_terms = []

            for other_term in terms:
                if term.term != other_term.term and term.term in other_term.term:
                    is_substring = True
                    containing_terms.append(other_term)

            if is_substring and containing_terms:
                # 含まれる用語がある場合
                max_container_freq = max([t.frequency for t in containing_terms])

                # 長い用語が相当数出現している場合、短い用語にペナルティ
                if max_container_freq > term.frequency * 0.10:  # 10%以上出現
                    original_score = term.score
                    # 頻度比に応じてペナルティを調整
                    freq_ratio = max_container_freq / term.frequency
                    if freq_ratio > 0.2:  # 20%以上の頻度比
                        term.score *= 0.2  # 強いペナルティ
                    else:
                        term.score *= 0.4  # 中程度のペナルティ

                    # デバッグ情報
                    if original_score > 0.2:  # 中程度以上のスコアを持つ用語
                        containing_info = ', '.join([f"{t.term}({t.frequency})" for t in containing_terms[:3]])
                        console.print(f"    [dim]ペナルティ: '{term.term}'({term.frequency}) → {original_score:.3f} × {term.score/original_score:.2f} = {term.score:.3f} [含む: {containing_info}][/dim]")

        # 複合語優先ルールを適用
        terms = self._prioritize_compound_terms(terms)

        return terms

    def _prioritize_compound_terms(self, terms: List[Term]) -> List[Term]:
        """
        同じベースを持つ用語群から最も具体的なものを優先

        Args:
            terms: 用語リスト

        Returns:
            優先度調整後の用語リスト
        """
        # 用語を長さでソート（長い順）
        sorted_terms = sorted(terms, key=lambda x: len(x.term), reverse=True)

        processed = set()
        for i, longer_term in enumerate(sorted_terms):
            if longer_term.term in processed:
                continue

            # この用語を含む短い用語を探す
            for j, shorter_term in enumerate(sorted_terms[i+1:], i+1):
                if shorter_term.term in longer_term.term and shorter_term.term != longer_term.term:
                    # 長い用語の頻度が十分にある場合
                    if longer_term.frequency > 3 and longer_term.frequency > shorter_term.frequency * 0.05:
                        # 短い用語に追加ペナルティ
                        penalty_factor = 0.5 if longer_term.frequency > shorter_term.frequency * 0.2 else 0.7
                        shorter_term.score *= penalty_factor
                        processed.add(shorter_term.term)

        return terms

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
        excluded_terms = []  # 除外された用語を記録
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
                elif content.startswith("NO"):
                    # 除外理由を記録
                    parts = content.split("|", 1)
                    reason = parts[1].strip() if len(parts) > 1 else "一般的な用語"
                    excluded_terms.append({
                        "term": term.term,
                        "score": term.score,
                        "frequency": term.frequency,
                        "reason": reason
                    })
                    
            except Exception as e:
                error_msg = str(e)
                if "content filter" in error_msg.lower():
                    console.print(f"[yellow]コンテンツフィルター ({term.term}): スキップ[/yellow]")
                    # コンテンツフィルターの場合はスコアベースで判断
                    if term.score > 0.5:  # スコアが高ければ含める
                        term.definition = f"{term.term}に関する用語"
                        validated_terms.append(term)
                else:
                    console.print(f"[yellow]検証エラー ({term.term}): {e}[/yellow]")
                    # その他のエラーの場合は含める（安全側に倒す）
                    term.definition = f"{term.term}（定義生成失敗）"
                    validated_terms.append(term)
        
        # 除外された用語を表示
        if excluded_terms:
            console.print(f"\n  [yellow]LLM検証で除外された用語: {len(excluded_terms)}件[/yellow]")
            console.print("  [dim]除外された上位10件:[/dim]")
            for i, excluded in enumerate(excluded_terms[:10]):
                console.print(f"    {i+1}. {excluded['term']}: スコア={excluded['score']:.3f}, 頻度={excluded['frequency']}, 理由={excluded['reason']}")

            # 全除外リストをファイルに保存
            if len(excluded_terms) > 10:
                console.print(f"    [dim]... 他 {len(excluded_terms) - 10}件[/dim]")

            # 除外リストをJSONファイルに保存
            import json
            from datetime import datetime

            excluded_file = Path(f"output/excluded_terms_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
            excluded_file.parent.mkdir(parents=True, exist_ok=True)

            with open(excluded_file, 'w', encoding='utf-8') as f:
                json.dump({
                    "total_excluded": len(excluded_terms),
                    "excluded_terms": excluded_terms
                }, f, ensure_ascii=False, indent=2)

            console.print(f"    [dim]除外リスト保存先: {excluded_file}[/dim]")

        console.print(f"  [green]検証後の専門用語数: {len(validated_terms)}件（除外: {len(excluded_terms)}件）[/green]")

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
        use_llm_validation=True,  # LLM検証を有効化
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
        console.print(f"\n[green]処理統計:[/green]")
        for key, value in stats.items():
            console.print(f"  {key}: {value}")

        # 処理段階の要約を表示
        console.print(f"\n[blue]処理段階のサマリー:[/blue]")
        console.print("  1. 形態素解析による候補抽出")
        console.print("  2. TF-IDF/C-valueスコア計算")
        console.print("  3. 最小頻度フィルタリング")
        if extractor.use_llm_validation:
            console.print("  4. LLM検証")
        console.print("  5. 上位30件を最終結果として選択")


if __name__ == "__main__":
    asyncio.run(main())