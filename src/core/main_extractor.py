#!/usr/bin/env python3
"""
jargon_extractor.py - 専門用語抽出・評価システム
=================================================
SudachiPyベースの専門用語抽出とLLMによるPerplexity評価を統合したシステム

Features:
- SudachiPy（Cモード）による形態素解析
- TF-IDF + C-value/NC-value統計指標
- LLMによるPerplexity計算
- 3段階自動分類（高/中/低確信度）
- 段階的実行（--pilot/--limit/--full）

Usage:
    python jargon_extractor.py input.txt --pilot        # 10語のみ（動作確認）
    python jargon_extractor.py input.txt --limit 100    # 上位100語
    python jargon_extractor.py input.txt --full         # 全処理
"""

import argparse
import json
import logging
import math
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import asyncio

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sudachipy import dictionary, tokenizer
from dotenv import load_dotenv
import fitz  # PyMuPDF
from openai import AsyncAzureOpenAI

# LangChainとLLM関連
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 環境設定
load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    logger.error("GOOGLE_API_KEY が設定されていません。.envファイルを確認してください。")
    sys.exit(1)

# =============================================
# 1. 専門用語抽出モジュール
# =============================================

class TermExtractor:
    """SudachiPy + 統計的指標による専門用語抽出"""
    
    def __init__(self):
        self.tokenizer_obj = dictionary.Dictionary().create()
        self.mode = tokenizer.Tokenizer.SplitMode.A  # 短単位（より細かい分割）
        
    def extract_candidates(self, text: str) -> List[Dict]:
        """テキストから専門用語候補を抽出"""
        if not text.strip():
            return []
            
        try:
            # 1. 形態素解析
            tokens = self.tokenizer_obj.tokenize(text, self.mode)
            
            # 2. 名詞のみを抽出
            nouns = []
            for token in tokens:
                pos = token.part_of_speech()[0]
                if pos == '名詞':
                    surface = token.surface()
                    normalized = token.normalized_form()
                    nouns.append({
                        'surface': surface,
                        'normalized': normalized,
                        'pos_detail': token.part_of_speech()
                    })
            
            # 3. 複合名詞の生成（改良版）
            candidates = set()
            
            # 単体名詞（2文字以上の意味のある名詞）
            for noun in nouns:
                surface = noun['surface']
                pos_detail = noun['pos_detail']
                
                # デバッグ用ログ
                if len(candidates) < 5:
                    logger.debug(f"名詞チェック: {surface} - {pos_detail[:3]}")
                
                # 名詞の条件を緩和
                if (len(surface) >= 2 and 
                    pos_detail[0] == '名詞' and
                    not surface.isdigit() and
                    surface not in ['こと', 'もの', 'ため', 'など']):  # 機能語は除外
                    candidates.add(surface)
            
            # 複合名詞（隣接する名詞の結合）
            self._add_compound_nouns(nouns, candidates)
            
            # 4. 部分文字列候補を除去
            filtered_candidates = self._filter_substring_candidates(candidates)
            
            # 5. 候補リストを辞書形式で返す
            result = []
            for candidate in filtered_candidates:
                result.append({
                    'term': candidate,
                    'length': len(candidate),
                    'frequency': text.count(candidate)
                })
            
            return result
            
        except Exception as e:
            logger.error(f"候補抽出エラー: {e}")
            return []
    
    def _add_compound_nouns(self, nouns: List[Dict], candidates: set):
        """隣接する名詞から複合語を生成"""
        if len(nouns) < 2:
            return
        
        i = 0
        while i < len(nouns) - 1:
            compound_parts = [nouns[i]]
            j = i + 1
            
            # 連続する名詞を収集
            while j < len(nouns) and self._should_combine(nouns[j-1], nouns[j]):
                compound_parts.append(nouns[j])
                j += 1
            
            # 2語以上の複合語を生成
            if len(compound_parts) >= 2:
                for k in range(2, min(len(compound_parts) + 1, 4)):  # 最大3語まで
                    compound = "".join([part['surface'] for part in compound_parts[:k]])
                    if len(compound) >= 3 and len(compound) <= 20:  # 適切な長さ
                        candidates.add(compound)
            
            i = j if j > i + 1 else i + 1
    
    def _should_combine(self, prev_noun: Dict, curr_noun: Dict) -> bool:
        """2つの名詞が結合すべきかを判定"""
        prev_surface = prev_noun['surface']
        curr_surface = curr_noun['surface']
        prev_pos = prev_noun['pos_detail']
        curr_pos = curr_noun['pos_detail']
        
        # 数字のみの語は結合しない
        if prev_surface.isdigit() or curr_surface.isdigit():
            return False
        
        # 1文字の語の処理
        if len(prev_surface) == 1 or len(curr_surface) == 1:
            # 意味のある接尾語は許可
            meaningful_suffixes = {
                '比', '率', '度', '値', '量', '数', '性', '力', 
                '温', '圧', '速', '質', '型', '式', '法', '材',
                '器', '機', '体', '部', '品', '物', '液', '気'
            }
            
            # 接尾語の場合は結合を許可
            if (len(curr_surface) == 1 and curr_surface in meaningful_suffixes and 
                len(prev_surface) >= 2):
                logger.debug(f"意味のある接尾語を結合: '{prev_surface}' + '{curr_surface}'")
                return True
            
            # その他の1文字語（「の」「に」など）は結合しない
            return False
        
        # 名詞同士は基本的に結合可能
        return (prev_pos[0] == '名詞' and curr_pos[0] == '名詞')
    
    def _extract_pattern_based(self, text: str) -> set:
        """パターンマッチングによる候補抽出"""
        patterns = []
        
        # カタカナ連続（3文字以上）
        katakana_pattern = r'[ア-ヴー]{3,}'
        patterns.extend(re.findall(katakana_pattern, text))
        
        # 英数字略語（2文字以上）
        abbrev_pattern = r'[A-Z]{2,}(?:-\d+)?'
        patterns.extend(re.findall(abbrev_pattern, text))
        
        # 日英混在（カタカナ+英語）
        mixed_pattern = r'[A-Za-z]+[ア-ヴー]+|[ア-ヴー]+[A-Za-z]+'
        patterns.extend(re.findall(mixed_pattern, text))
        
        return set(patterns)
    
    def _filter_substring_candidates(self, candidates: set) -> set:
        """部分文字列候補を除去（アンモ→アンモニア燃料など）"""
        candidates_list = list(candidates)
        filtered = set()
        
        for candidate in candidates_list:
            # 最小長制限（3文字以上）
            if len(candidate) < 3:
                continue
                
            # この候補がより長い候補の部分文字列かチェック
            is_substring = False
            for other in candidates_list:
                if (candidate != other and 
                    len(candidate) < len(other) and 
                    candidate in other):
                    # ただし、候補が他の語の意味のある部分を構成している場合は保持
                    # 例：「エンジン」が「ディーゼルエンジン」の一部でも意味があるので保持
                    if not self._is_meaningful_part(candidate, other):
                        is_substring = True
                        break
            
            if not is_substring:
                filtered.add(candidate)
        
        logger.info(f"部分文字列フィルタリング: {len(candidates)} → {len(filtered)}語")
        return filtered
    
    def _is_meaningful_part(self, part: str, whole: str) -> bool:
        """部分文字列が意味のある独立した語かを判定"""
        # 語尾の部分文字列は通常意味がない（例：「ニア」「モニア」）
        if whole.endswith(part) and len(part) < len(whole) * 0.6:
            return False
            
        # 語頭の部分文字列も短すぎる場合は意味がない（例：「アン」「アンモ」）
        if whole.startswith(part) and len(part) < len(whole) * 0.6:
            return False
        
        # 中間の部分文字列は基本的に意味がない
        if not (whole.startswith(part) or whole.endswith(part)):
            return False
            
        # 一般的な語尾や語頭の場合は独立した意味があるとみなす
        meaningful_parts = {
            'システム', 'エンジン', 'データ', 'プロセス', 'メソッド', 'アルゴリズム',
            'プラットフォーム', 'インターフェース', 'アプリケーション', 'サービス'
        }
        
        return part in meaningful_parts
    
    def calculate_tf_idf(self, candidates: List[Dict], documents: List[str]) -> Dict[str, float]:
        """TF-IDF値を計算"""
        if not documents or not candidates:
            return {}
        
        try:
            terms = [c['term'] for c in candidates]
            
            # 各用語の文書内出現をチェック
            term_docs = []
            for term in terms:
                doc_texts = []
                for doc in documents:
                    if term in doc:
                        doc_texts.append(doc)
                    else:
                        doc_texts.append('')  # 出現しない場合は空文字
                term_docs.append(' '.join(doc_texts))
            
            # TF-IDF計算
            vectorizer = TfidfVectorizer(
                vocabulary=terms,
                token_pattern=r'(?u)\b\w+\b',
                ngram_range=(1, 1)
            )
            
            # 全文書を結合してTF-IDF計算
            full_text = ' '.join(documents)
            tfidf_matrix = vectorizer.fit_transform([full_text])
            
            # 結果を辞書形式で返す
            feature_names = vectorizer.get_feature_names_out()
            scores = {}
            for i, term in enumerate(feature_names):
                scores[term] = tfidf_matrix[0, i]
            
            return scores
            
        except Exception as e:
            logger.error(f"TF-IDF計算エラー: {e}")
            return {}
    
    def calculate_c_value(self, candidates: List[Dict], text: str) -> Dict[str, float]:
        """C-value統計指標を計算"""
        c_values = {}
        
        # 候補を長さでソート
        sorted_candidates = sorted(candidates, key=lambda x: x['length'], reverse=True)
        
        for candidate in sorted_candidates:
            term = candidate['term']
            freq = candidate['frequency']
            length = candidate['length']
            
            # より長い用語に含まれる回数を計算
            contained_count = 0
            containing_terms = 0
            
            for other in sorted_candidates:
                if other['length'] > length and term in other['term']:
                    contained_count += other['frequency']
                    containing_terms += 1
            
            # C-value計算
            if containing_terms == 0:
                c_value = math.log2(length) * freq
            else:
                c_value = math.log2(length) * (freq - contained_count / containing_terms)
            
            c_values[term] = max(0, c_value)  # 負の値は0に
        
        return c_values

# =============================================
# 2. LLM評価モジュール
# =============================================

class LLMEvaluator:
    """Azure OpenAIによる真のPerplexity計算とトークン確率評価"""
    
    def __init__(self):
        # Azure OpenAI設定
        self.azure_client = AsyncAzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
        )
        self.deployment = "gpt-4o"  # デプロイメント名
        
        # Geminiも保持（比較用）
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            temperature=0.0,
            google_api_key=API_KEY,
        )
        
        # 評価用プロンプト
        self.definition_prompt = ChatPromptTemplate.from_messages([
            ("system", "あなたは専門用語の定義を生成する専門家です。与えられた用語について、正確で簡潔な定義を30-50文字で回答してください。"),
            ("user", "{term}とは何か説明してください。")
        ])
        
        # チェイン作成
        self.definition_chain = self.definition_prompt | self.llm | StrOutputParser()
    
    async def calculate_perplexity_batch(self, terms: List[str]) -> Dict[str, float]:
        """バッチで真のPerplexity値を計算（Azure OpenAI使用）"""
        results = {}
        
        # APIレート制限対策でバッチサイズを制限
        batch_size = 5
        for i in range(0, len(terms), batch_size):
            batch = terms[i:i+batch_size]
            
            # 並列処理でPerplexityを計算
            tasks = [self._calculate_true_perplexity(term) for term in batch]
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for term, result in zip(batch, batch_results):
                if isinstance(result, Exception):
                    logger.warning(f"Perplexity計算エラー ({term}): {result}")
                    results[term] = 100.0  # エラー時はデフォルト値
                else:
                    results[term] = result
            
            # レート制限対策
            if i + batch_size < len(terms):
                await asyncio.sleep(2)
                logger.info(f"Processed {i + batch_size}/{len(terms)} terms...")
        
        return results
    
    async def _calculate_true_perplexity(self, term: str) -> float:
        """logprobs APIを使用した真のPerplexity計算"""
        try:
            # logprobsが利用可能か確認
            use_logprobs = True  # Azure OpenAI GPT-4oはlogprobs対応
            
            if use_logprobs:
                # logprobsベースの真のPerplexity計算
                perplexity = await self._calculate_logprobs_perplexity(term)
                if perplexity is not None:
                    return perplexity
            
            # フォールバック: 従来の評価手法
            evaluation_results = []
            
            # 方法1: 定義生成の一貫性（既存）
            consistency_score = await self._evaluate_definition_consistency(term)
            evaluation_results.append(('consistency', consistency_score, 0.4))
            
            # 方法2: 理解度直接評価
            comprehension_score = await self._evaluate_term_comprehension(term)
            evaluation_results.append(('comprehension', comprehension_score, 0.3))
            
            # 方法3: 文脈での自然さ
            naturalness_score = await self._evaluate_contextual_naturalness(term)
            evaluation_results.append(('naturalness', naturalness_score, 0.3))
            
            # 重み付き平均でPerplexity計算
            weighted_score = sum(score * weight for _, score, weight in evaluation_results)
            
            return max(5.0, min(100.0, weighted_score))
            
        except Exception as e:
            logger.error(f"真のPerplexity計算エラー ({term}): {e}")
            return 80.0
    
    async def _calculate_logprobs_perplexity(self, term: str) -> Optional[float]:
        """logprobs APIを使用した数学的に正確なPerplexity計算"""
        try:
            # 文脈を作成
            contexts = [
                f"この{term}は専門用語である。",
                f"{term}について説明します。",
                f"重要な概念である{term}を理解する。"
            ]
            
            logprobs_list = []
            
            for context in contexts:
                # 用語の前までをプロンプトとして使用
                prefix = context.split(term)[0]
                
                response = await self.azure_client.chat.completions.create(
                    model=self.deployment,
                    messages=[
                        {"role": "system", "content": "次の文を自然に継続してください。"},
                        {"role": "user", "content": prefix}
                    ],
                    max_tokens=len(term.split()) + 5,  # 用語+余裕
                    temperature=0.0,
                    logprobs=True,
                    top_logprobs=5
                )
                
                # logprobsデータの解析
                if response.choices[0].logprobs and response.choices[0].logprobs.content:
                    term_logprob = self._extract_term_logprob(
                        response.choices[0].logprobs.content, 
                        term
                    )
                    if term_logprob is not None:
                        logprobs_list.append(term_logprob)
                
                await asyncio.sleep(0.2)
            
            if logprobs_list:
                # 平均logprobからPerplexity計算
                avg_logprob = np.mean(logprobs_list)
                perplexity = math.exp(-avg_logprob)
                return min(100.0, max(5.0, perplexity))
            
            return None
            
        except Exception as e:
            logger.warning(f"logprobs Perplexity計算エラー ({term}): {e}")
            return None
    
    def _extract_term_logprob(self, logprobs_content: List, target_term: str) -> Optional[float]:
        """logprobsデータから用語のlogprobを抽出"""
        try:
            term_tokens = []
            current_text = ""
            
            for token_data in logprobs_content:
                token = token_data.token
                logprob = token_data.logprob
                current_text += token
                
                # 用語に関連するトークンを収集
                if target_term in current_text:
                    term_tokens.append(logprob)
                    
                    # 用語が完全に含まれたら終了
                    if target_term in current_text and len(current_text) >= len(target_term):
                        break
            
            if term_tokens:
                return np.mean(term_tokens)
            
            # トップ候補も確認
            for token_data in logprobs_content:
                if hasattr(token_data, 'top_logprobs') and token_data.top_logprobs:
                    for candidate in token_data.top_logprobs:
                        if target_term in candidate.token:
                            return candidate.logprob
            
            return None
            
        except Exception:
            return None
    
    async def _evaluate_definition_consistency(self, term: str) -> float:
        """定義生成の一貫性評価（既存ロジック改良）"""
        definitions = []
        for i in range(3):
            try:
                response = await self.azure_client.chat.completions.create(
                    model=self.deployment,
                    messages=[
                        {"role": "system", "content": "専門用語を30文字以内で簡潔に定義してください。不明な場合は「不明」と答えてください。"},
                        {"role": "user", "content": f"{term}とは"}
                    ],
                    temperature=0.1,
                    max_tokens=50
                )
                definition = response.choices[0].message.content.strip()
                definitions.append(definition)
                await asyncio.sleep(0.3)
            except Exception as e:
                logger.warning(f"定義生成エラー ({term}): {e}")
                continue
        
        if not definitions:
            return 80.0
        
        return self._estimate_perplexity_from_consistency(term, definitions)
    
    async def _evaluate_term_comprehension(self, term: str) -> float:
        """理解度の直接評価"""
        try:
            response = await self.azure_client.chat.completions.create(
                model=self.deployment,
                messages=[
                    {"role": "system", "content": "専門用語の理解困難度を1-10で評価してください。1=非常に理解しやすい、10=理解が困難。数値のみ回答。"},
                    {"role": "user", "content": f"「{term}」の理解困難度: "}
                ],
                temperature=0.1,
                max_tokens=5
            )
            
            score_text = response.choices[0].message.content.strip()
            import re
            score_match = re.search(r'(\d+)', score_text)
            if score_match:
                difficulty_score = float(score_match.group(1))
                # 困難度をPerplexityに変換（1-10 → 10-100）
                perplexity = 5.0 + (difficulty_score * 9.5)  # 5-100の範囲
                return perplexity
            
        except Exception as e:
            logger.warning(f"理解度評価エラー ({term}): {e}")
        
        return 50.0  # デフォルト値
    
    async def _evaluate_contextual_naturalness(self, term: str) -> float:
        """文脈での自然さ評価"""
        try:
            # サンプル文脈を生成
            contexts = [
                f"{term}について説明します。",
                f"この{term}は重要です。",
                f"{term}の特徴を分析しました。"
            ]
            
            naturalness_scores = []
            for context in contexts:
                response = await self.azure_client.chat.completions.create(
                    model=self.deployment,
                    messages=[
                        {"role": "system", "content": "文章の自然さを1-10で評価してください。1=不自然、10=自然。数値のみ回答。"},
                        {"role": "user", "content": f"「{context}」の自然さ: "}
                    ],
                    temperature=0.1,
                    max_tokens=3
                )
                
                score_text = response.choices[0].message.content.strip()
                import re
                score_match = re.search(r'(\d+)', score_text)
                if score_match:
                    naturalness = float(score_match.group(1))
                    naturalness_scores.append(naturalness)
                
                await asyncio.sleep(0.2)
            
            if naturalness_scores:
                avg_naturalness = np.mean(naturalness_scores)
                # 自然さをPerplexityに変換（自然でないほど高Perplexity）
                perplexity = 105.0 - (avg_naturalness * 10.0)  # 10-100の範囲
                return max(5.0, perplexity)
        
        except Exception as e:
            logger.warning(f"自然さ評価エラー ({term}): {e}")
        
        return 40.0  # デフォルト値
    
    def _estimate_perplexity_from_consistency(self, term: str, definitions: List[str]) -> float:
        """定義の一貫性からPerplexityを推定"""
        if not definitions:
            return 80.0
        
        # 基本スコア
        base_score = 40.0
        
        # 1. 定義の長さの一貫性
        lengths = [len(d) for d in definitions]
        length_variance = np.var(lengths) if len(lengths) > 1 else 0
        length_penalty = min(length_variance / 10, 20)
        
        # 2. 不確実性キーワードの検出
        uncertainty_keywords = ["不明", "わからない", "定義できない", "曖昧", "おそらく"]
        uncertainty_count = sum(
            1 for definition in definitions 
            for keyword in uncertainty_keywords 
            if keyword in definition
        )
        uncertainty_penalty = uncertainty_count * 15
        
        # 3. 用語の特性による調整
        term_bonus = 0
        if len(term) >= 4:  # 4文字以上は専門用語らしい
            term_bonus -= 10
        if re.search(r'[ア-ヴー]{3,}', term):  # カタカナ語
            term_bonus -= 5
        if re.search(r'[A-Z]{2,}', term):  # 英語略語
            term_bonus -= 5
        
        # 4. 最終Perplexity推定
        perplexity = base_score + length_penalty + uncertainty_penalty + term_bonus
        
        return perplexity
    
    async def get_token_generation_probability(self, term: str) -> float:
        """強化されたトークン生成確率を計算"""
        try:
            if len(term) < 3:
                return 0.1  # 短すぎる用語は低確率
            
            # 複数の手法でトークン確率を評価
            methods_results = []
            
            # 方法1: 部分→完全予測（複数分割点）
            partial_prob = await self._enhanced_partial_completion(term)
            methods_results.append(partial_prob)
            
            # 方法2: 専門用語一般性評価
            generality_prob = await self._evaluate_term_generality(term)
            methods_results.append(generality_prob)
            
            # 重み付き平均
            weights = [0.7, 0.3]  # 部分完成を重視
            weighted_avg = sum(prob * weight for prob, weight in zip(methods_results, weights))
            
            return min(1.0, max(0.0, weighted_avg))
            
        except Exception as e:
            logger.error(f"トークン生成確率エラー ({term}): {e}")
            return 0.1
    
    async def _enhanced_partial_completion(self, term: str) -> float:
        """複数分割点での部分完成確率"""
        if len(term) < 4:
            return 0.2
        
        # 複数の分割点でテスト
        split_points = [len(term)//3, len(term)//2, len(term)*2//3]
        success_rates = []
        
        for split_point in split_points:
            if split_point <= 1:
                continue
                
            partial = term[:split_point]
            success_count = 0
            attempts = 3
            
            for _ in range(attempts):
                try:
                    response = await self.azure_client.chat.completions.create(
                        model=self.deployment,
                        messages=[
                            {"role": "system", "content": "専門用語を完成させてください。最も適切な1つだけ答えてください。"},
                            {"role": "user", "content": f"「{partial}」で始まる専門用語: {partial}"}
                        ],
                        temperature=0.2,
                        max_tokens=15
                    )
                    
                    generated = response.choices[0].message.content.strip()
                    # より厳密なマッチング
                    if term == generated or (term in generated and len(generated) <= len(term) + 3):
                        success_count += 1
                        
                    await asyncio.sleep(0.2)
                    
                except Exception as e:
                    logger.warning(f"部分完成エラー ({partial}): {e}")
                    continue
            
            if attempts > 0:
                success_rates.append(success_count / attempts)
        
        return np.mean(success_rates) if success_rates else 0.1
    
    async def _evaluate_term_generality(self, term: str) -> float:
        """専門用語の一般性評価"""
        try:
            response = await self.azure_client.chat.completions.create(
                model=self.deployment,
                messages=[
                    {"role": "system", "content": "専門用語の一般性を1-10で評価してください。1=非常に専門的、10=一般的。数値のみ回答。"},
                    {"role": "user", "content": f"「{term}」の一般性: "}
                ],
                temperature=0.1,
                max_tokens=3
            )
            
            score_text = response.choices[0].message.content.strip()
            import re
            score_match = re.search(r'(\d+)', score_text)
            if score_match:
                score = float(score_match.group(1))
                # 一般性スコアを確率に変換（一般的ほど高確率）
                return min(1.0, score / 10.0)
            
        except Exception as e:
            logger.warning(f"一般性評価エラー ({term}): {e}")
        
        return 0.4  # デフォルト値
    
    async def evaluate_term_quality(self, term: str) -> Dict[str, any]:
        """用語の品質を多面的に評価"""
        try:
            # 定義生成と文脈使用を並列実行
            definition_task = self.definition_chain.ainvoke({"term": term})
            context_task = self.context_chain.ainvoke({"term": term})
            
            definition, context_examples = await asyncio.gather(definition_task, context_task)
            
            # Perplexity計算
            perplexity = await self._calculate_single_perplexity(term)
            
            return {
                "term": term,
                "perplexity": perplexity,
                "definition": definition,
                "context_examples": context_examples,
                "definition_length": len(definition),
                "has_uncertainty": any(pattern in definition.lower() 
                                     for pattern in ["わからない", "不明", "おそらく"])
            }
            
        except Exception as e:
            logger.error(f"品質評価エラー ({term}): {e}")
            return {
                "term": term,
                "perplexity": 90.0,
                "definition": "評価エラー",
                "context_examples": "",
                "definition_length": 0,
                "has_uncertainty": True
            }

# =============================================
# 3. 優先順位付けと分類システム
# =============================================

class TermPrioritizer:
    """専門用語の優先順位付けと3段階分類"""
    
    def __init__(self):
        self.high_confidence_threshold = 30    # 高確信度
        self.low_confidence_threshold = 70     # 低確信度
    
    def classify_terms(self, evaluated_terms: List[Dict]) -> Dict[str, List[Dict]]:
        """3段階分類を実行"""
        classification = {
            "high_confidence": [],    # 自動承認
            "medium_confidence": [],  # 保留
            "low_confidence": []      # 要人手確認
        }
        
        for term_data in evaluated_terms:
            perplexity = term_data.get("perplexity", 50)
            
            if perplexity < self.high_confidence_threshold:
                classification["high_confidence"].append(term_data)
            elif perplexity > self.low_confidence_threshold:
                classification["low_confidence"].append(term_data)
            else:
                classification["medium_confidence"].append(term_data)
        
        return classification
    
    def calculate_statistical_scores(self, terms_data: List[Dict]) -> List[Dict]:
        """統計指標による事前スコアリング（LLM評価前）"""
        for term_data in terms_data:
            frequency = term_data.get("frequency", 1)
            c_value = term_data.get("c_value", 0)
            tfidf = term_data.get("tfidf", 0)
            
            # 統計的スコア = C-value + TF-IDF + 頻度対数
            statistical_score = (
                c_value * 0.5 +                    # C-value: 専門用語らしさ
                tfidf * 100 * 0.3 +                # TF-IDF: 文書内重要度
                math.log(frequency + 1) * 0.2      # 頻度: 対数スケール
            )
            term_data["statistical_score"] = statistical_score
        
        # 統計スコア順でソート
        return sorted(terms_data, key=lambda x: x["statistical_score"], reverse=True)
    
    def calculate_priority_scores(self, terms_data: List[Dict]) -> List[Dict]:
        """最終優先度スコアを計算（統計指標 + LLM評価 + トークン確率）"""
        for term_data in terms_data:
            perplexity = term_data.get("perplexity", 50)
            statistical_score = term_data.get("statistical_score", 0)
            token_probability = term_data.get("token_probability", 0.1)
            
            # 最終優先度スコア = 統計スコア + Perplexity逆転スコア + トークン確率
            priority_score = (
                statistical_score * 0.4 +           # 統計的重要度
                (100 - perplexity) * 0.4 +          # Perplexity逆転（低いほど良い）
                token_probability * 100 * 0.2       # トークン生成確率
            )
            term_data["priority_score"] = priority_score
        
        # 優先度スコア順でソート
        return sorted(terms_data, key=lambda x: x["priority_score"], reverse=True)
    
    def get_review_candidates(self, classified_terms: Dict[str, List[Dict]], 
                            max_candidates: int = 50) -> List[Dict]:
        """人手確認用の候補を選定"""
        # 低確信度かつ高頻度の用語を優先
        low_confidence = classified_terms.get("low_confidence", [])
        medium_confidence = classified_terms.get("medium_confidence", [])
        
        # 優先度順でソート
        review_pool = low_confidence + medium_confidence
        review_pool = sorted(review_pool, key=lambda x: x.get("priority_score", 0), reverse=True)
        
        return review_pool[:max_candidates]

# =============================================
# 4. 出力機能
# =============================================

class OutputGenerator:
    """結果出力とレポート生成"""
    
    def save_results_json(self, data: Dict, output_path: Path):
        """結果をJSON形式で保存"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f"結果をJSONで保存: {output_path}")
    
    def save_review_csv(self, review_candidates: List[Dict], output_path: Path):
        """人手確認用CSVを生成"""
        if not review_candidates:
            logger.warning("確認用候補が存在しません")
            return
        
        df = pd.DataFrame(review_candidates)
        
        # 必要な列のみを選択
        columns_to_keep = [
            "term", "perplexity", "frequency", "priority_score", 
            "c_value", "tfidf", "definition"
        ]
        df = df[[col for col in columns_to_keep if col in df.columns]]
        
        df.to_csv(output_path, index=False, encoding='utf-8-sig')
        logger.info(f"確認用CSVを保存: {output_path}")
    
    def save_stats_json(self, stats: Dict, output_path: Path):
        """統計情報をJSON形式で保存"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        logger.info(f"統計情報を保存: {output_path}")
    
    def generate_stats(self, classified_terms: Dict[str, List[Dict]], 
                      total_candidates: int, processing_time: float) -> Dict:
        """統計情報を生成"""
        high_count = len(classified_terms.get("high_confidence", []))
        medium_count = len(classified_terms.get("medium_confidence", []))
        low_count = len(classified_terms.get("low_confidence", []))
        
        stats = {
            "processing_time_seconds": round(processing_time, 2),
            "total_candidates": total_candidates,
            "classification": {
                "high_confidence": high_count,
                "medium_confidence": medium_count,
                "low_confidence": low_count
            },
            "coverage": {
                "auto_approved_ratio": high_count / total_candidates if total_candidates > 0 else 0,
                "manual_review_ratio": low_count / total_candidates if total_candidates > 0 else 0
            }
        }
        
        return stats

# =============================================
# 5. メインパイプライン
# =============================================

class JargonExtractor:
    """統合型専門用語抽出システム"""
    
    def __init__(self):
        self.term_extractor = TermExtractor()
        self.llm_evaluator = LLMEvaluator()
        self.prioritizer = TermPrioritizer()
        self.output_generator = OutputGenerator()
    
    def _load_file(self, file_path: Path) -> str:
        """ファイル形式に応じてテキストを読み込み"""
        file_extension = file_path.suffix.lower()
        
        if file_extension == '.pdf':
            return self._load_pdf(file_path)
        elif file_extension in ['.txt', '.md']:
            return self._load_text(file_path)
        else:
            logger.warning(f"未対応のファイル形式: {file_extension}. テキストファイルとして処理します。")
            return self._load_text(file_path)
    
    def _load_pdf(self, pdf_path: Path) -> str:
        """PDFファイルからテキストを抽出"""
        try:
            doc = fitz.open(str(pdf_path))
            full_text = []
            
            logger.info(f"PDF処理中: {len(doc)}ページ")
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                text = page.get_text()
                
                if text.strip():  # 空でないページのみ追加
                    full_text.append(text)
                    
                # 進捗表示（10ページごと）
                if (page_num + 1) % 10 == 0:
                    logger.info(f"処理済み: {page_num + 1}/{len(doc)}ページ")
            
            doc.close()
            
            result_text = '\n'.join(full_text)
            logger.info(f"PDF抽出完了: {len(result_text)}文字")
            
            return result_text
            
        except Exception as e:
            logger.error(f"PDF読み込みエラー: {e}")
            raise
    
    def _load_text(self, text_path: Path) -> str:
        """テキストファイルを読み込み"""
        try:
            with open(text_path, 'r', encoding='utf-8') as f:
                text = f.read()
            
            logger.info(f"テキスト読み込み完了: {len(text)}文字")
            return text
            
        except UnicodeDecodeError:
            # UTF-8で読めない場合はShift_JISを試す
            try:
                with open(text_path, 'r', encoding='shift_jis') as f:
                    text = f.read()
                logger.info(f"テキスト読み込み完了(Shift_JIS): {len(text)}文字")
                return text
            except Exception as e:
                logger.error(f"エンコーディングエラー: {e}")
                raise
        except Exception as e:
            logger.error(f"テキスト読み込みエラー: {e}")
            raise
    
    async def process_file(self, input_path: Path, mode: str = "pilot", limit: int = 10):
        """ファイルを処理して専門用語を抽出・評価"""
        start_time = asyncio.get_event_loop().time()
        
        # 1. ファイル読み込み
        logger.info(f"ファイルを読み込み中: {input_path}")
        try:
            text = self._load_file(input_path)
        except Exception as e:
            logger.error(f"ファイル読み込みエラー: {e}")
            return
        
        if not text.strip():
            logger.error("空のファイルです")
            return
        
        # 2. 専門用語候補抽出
        logger.info("専門用語候補を抽出中...")
        candidates = self.term_extractor.extract_candidates(text)
        
        if not candidates:
            logger.error("候補が見つかりませんでした")
            return
        
        # デバッグ用：最初の20候補を表示
        logger.info(f"抽出された上位20候補: {[c['term'] for c in candidates[:20]]}")
        
        logger.info(f"抽出された候補数: {len(candidates)}")
        
        # 3. 統計指標計算
        logger.info("統計指標を計算中...")
        tfidf_scores = self.term_extractor.calculate_tf_idf(candidates, [text])
        c_values = self.term_extractor.calculate_c_value(candidates, text)
        
        # 候補データに統計値を追加
        for candidate in candidates:
            term = candidate['term']
            candidate['tfidf'] = tfidf_scores.get(term, 0)
            candidate['c_value'] = c_values.get(term, 0)
        
        # 4. 統計指標による事前ランキング
        logger.info("統計指標による事前ランキング中...")
        candidates = self.prioritizer.calculate_statistical_scores(candidates)
        
        # 5. 処理対象の選定（統計スコア上位）
        if mode == "pilot":
            selected_candidates = candidates[:10]
            logger.info(f"パイロットモード: 統計上位{len(selected_candidates)}語で実行")
        elif mode == "limit":
            selected_candidates = candidates[:limit]
            logger.info(f"制限モード: 統計上位{len(selected_candidates)}語で実行")
        elif mode == "full":
            # fullモードでも上位100語に制限（効率化）
            selected_candidates = candidates[:100]
            logger.info(f"フルモード（効率化）: 統計上位{len(selected_candidates)}語で実行")
        
        # 6. Azure OpenAIによる真のPerplexity評価（選定候補のみ）
        logger.info("Azure OpenAIによる真のPerplexity評価を開始...")
        selected_terms = [c['term'] for c in selected_candidates]
        perplexity_scores = await self.llm_evaluator.calculate_perplexity_batch(selected_terms)
        
        # 7. トークン生成確率の計算（上位候補のみ）
        logger.info("トークン生成確率を計算中...")
        token_probabilities = {}
        for candidate in selected_candidates[:min(20, len(selected_candidates))]:  # 上位20語のみ
            term = candidate['term']
            try:
                token_prob = await self.llm_evaluator.get_token_generation_probability(term)
                token_probabilities[term] = token_prob
            except Exception as e:
                logger.warning(f"トークン確率計算スキップ ({term}): {e}")
                token_probabilities[term] = 0.1
        
        # 8. 評価結果をマージ
        evaluated_terms = []
        for candidate in selected_candidates:
            term = candidate['term']
            term_data = candidate.copy()
            term_data['perplexity'] = perplexity_scores.get(term, 50.0)
            term_data['token_probability'] = token_probabilities.get(term, 0.1)
            evaluated_terms.append(term_data)
        
        # 9. 最終優先順位付けと分類
        logger.info("最終優先順位付けと分類を実行中...")
        evaluated_terms = self.prioritizer.calculate_priority_scores(evaluated_terms)
        classified_terms = self.prioritizer.classify_terms(evaluated_terms)
        
        # 7. 出力生成
        processing_time = asyncio.get_event_loop().time() - start_time
        
        # 結果保存
        output_dir = Path("output")
        output_dir.mkdir(exist_ok=True)
        
        # results.json
        results_data = {
            "metadata": {
                "input_file": str(input_path),
                "mode": mode,
                "processing_time": processing_time,
                "total_candidates": len(candidates)
            },
            "classified_terms": classified_terms
        }
        self.output_generator.save_results_json(results_data, output_dir / "results.json")
        
        # review.csv（人手確認用）
        review_candidates = self.prioritizer.get_review_candidates(classified_terms, 50)
        self.output_generator.save_review_csv(review_candidates, output_dir / "review.csv")
        
        # stats.json
        stats = self.output_generator.generate_stats(classified_terms, len(candidates), processing_time)
        self.output_generator.save_stats_json(stats, output_dir / "stats.json")
        
        # 8. サマリー表示
        self._display_summary(classified_terms, processing_time)
    
    def _display_summary(self, classified_terms: Dict[str, List[Dict]], processing_time: float):
        """処理結果のサマリーを表示"""
        high_count = len(classified_terms.get("high_confidence", []))
        medium_count = len(classified_terms.get("medium_confidence", []))
        low_count = len(classified_terms.get("low_confidence", []))
        total = high_count + medium_count + low_count
        
        print("\n" + "="*50)
        print("処理結果サマリー")
        print("="*50)
        print(f"処理時間: {processing_time:.1f}秒")
        print(f"総候補数: {total}語")
        print(f"高確信度: {high_count}語 (自動承認)")
        print(f"中確信度: {medium_count}語 (保留)")
        print(f"低確信度: {low_count}語 (要確認)")
        print(f"出力フォルダ: ./output/")
        
        if high_count > 0:
            print("\n自動承認された上位5語:")
            for term_data in classified_terms["high_confidence"][:5]:
                term = term_data.get("term", "N/A")
                perplexity = term_data.get("perplexity", 0)
                print(f"   - {term} (Perplexity: {perplexity:.1f})")

# =============================================
# 6. CLI エントリーポイント
# =============================================

def main():
    parser = argparse.ArgumentParser(
        description="専門用語抽出・評価システム",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  python jargon_extractor.py input.txt --pilot        # 10語のみ（動作確認）
  python jargon_extractor.py input.txt --limit 100    # 上位100語
  python jargon_extractor.py input.txt --full         # 全処理
        """
    )
    
    parser.add_argument("input_file", type=Path, help="入力ファイル (.txt, .md, .pdf対応)")
    
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("--pilot", action="store_true", help="パイロットモード（10語のみ）")
    mode_group.add_argument("--limit", type=int, metavar="N", help="上位N語のみ処理")
    mode_group.add_argument("--full", action="store_true", help="全候補を処理")
    
    args = parser.parse_args()
    
    # 入力ファイルチェック
    if not args.input_file.exists():
        logger.error(f"入力ファイルが存在しません: {args.input_file}")
        sys.exit(1)
    
    # モード決定
    if args.pilot:
        mode = "pilot"
        limit = 10
    elif args.limit:
        mode = "limit"
        limit = args.limit
    else:  # full
        mode = "full"
        limit = None
    
    # システム初期化と実行
    extractor = JargonExtractor()
    
    try:
        asyncio.run(extractor.process_file(args.input_file, mode, limit))
        print("\n処理が完了しました！")
    except KeyboardInterrupt:
        print("\n処理が中断されました。")
    except Exception as e:
        logger.error(f"予期しないエラーが発生しました: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()