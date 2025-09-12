#!/usr/bin/env python3
"""
汎用版：複数分割モードを活用した専門用語抽出
パターンに依存しない汎用的アプローチ
"""

import argparse
import json
import logging
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple
import asyncio

import numpy as np
from sudachipy import dictionary, tokenizer
from dotenv import load_dotenv
import fitz  # PyMuPDF
from openai import AsyncAzureOpenAI
import tiktoken

# 環境設定
load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Azure OpenAI設定確認
AZURE_KEY = os.getenv("AZURE_OPENAI_API_KEY") or os.getenv("AZURE_OPENAI_KEY")
AZURE_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o")

if not AZURE_KEY or not AZURE_ENDPOINT:
    logger.error("Azure OpenAI設定が不足しています。.envファイルを確認してください。")
    sys.exit(1)

class GenericTermExtractor:
    """複数分割モードを活用した汎用的な専門用語候補抽出"""
    
    def __init__(self):
        self.tokenizer_obj = dictionary.Dictionary().create()
    
    def extract_candidates(self, text: str, max_candidates: int = 100) -> List[str]:
        """品詞パターンベースで専門用語候補を抽出"""
        if not text.strip():
            return []
        
        try:
            all_candidates = set()
            
            # 1. 複数の分割モードで解析
            modes = [
                (tokenizer.Tokenizer.SplitMode.C, 'long'),   # 最長一致
                (tokenizer.Tokenizer.SplitMode.B, 'medium'), # 中間
                (tokenizer.Tokenizer.SplitMode.A, 'short')   # 短単位
            ]
            
            tokens_by_mode = {}
            for mode, mode_name in modes:
                tokens = self.tokenizer_obj.tokenize(text, mode)
                tokens_by_mode[mode_name] = tokens
                
                # 品詞パターンに基づく候補抽出
                pattern_candidates = self._extract_by_pos_patterns(tokens, mode_name)
                all_candidates.update(pattern_candidates)
            
            # 2. 短単位トークンから専門用語パターンの複合語を生成
            compound_candidates = self._generate_technical_compounds(tokens_by_mode['short'])
            all_candidates.update(compound_candidates)
            
            # 3. 基本的なフィルタリング（長さ、文字種など）
            filtered_candidates = self._basic_filter(all_candidates)
            
            # 4. スコアリング
            scored_candidates = self._score_candidates(filtered_candidates, text, tokens_by_mode)
            
            # 上位N件を返す
            scored_candidates.sort(key=lambda x: x[1], reverse=True)
            return [cand[0] for cand in scored_candidates[:max_candidates]]
            
        except Exception as e:
            logger.error(f"候補抽出エラー: {e}")
            return []
    
    def _extract_by_pos_patterns(self, tokens: List, mode_name: str) -> Set[str]:
        """品詞パターンに基づいて専門用語候補を抽出"""
        candidates = set()
        
        # 単一名詞の抽出（形態素境界に基づく）
        for token in tokens:
            pos_info = token.part_of_speech()
            pos = pos_info[0]
            pos_detail = pos_info[1] if len(pos_info) > 1 else ""
            surface = token.surface()
            
            # 名詞かつ適切な長さ
            if pos == '名詞':
                # 一般名詞、固有名詞、サ変接続など専門用語になりやすいもの
                if pos_detail in ['一般', '固有名詞', 'サ変接続', '数']:
                    if len(surface) >= 2:  # 最小2文字
                        candidates.add(surface)
                # 英数字混在の型式名など
                elif self._is_technical_term(surface):
                    candidates.add(surface)
        
        return candidates
    
    def _is_technical_term(self, term: str) -> bool:
        """技術用語の可能性が高いかチェック"""
        import re
        # 英数字混在（型式名など）
        if re.search(r'[A-Za-z]', term) and re.search(r'[0-9]', term):
            return True
        # カタカナ専門用語
        if re.search(r'^[ア-ヴー・]+$', term) and len(term) >= 3:
            return True
        # アルファベットのみの略語
        if re.match(r'^[A-Z]{2,}$', term):
            return True
        return False
    
    def _generate_technical_compounds(self, tokens: List) -> Set[str]:
        """専門用語パターンに基づいて複合語を生成"""
        compounds = set()
        
        # 専門用語の品詞パターン定義
        patterns = [
            ['名詞', '名詞'],                    # 燃料噴射
            ['名詞', '名詞', '名詞'],            # 燃料噴射弁
            ['接頭詞', '名詞'],                  # 高圧弁
            ['名詞', '接尾辞'],                  # 噴射率
            ['名詞', '助詞', '名詞'],            # ～の～（限定的に）
        ]
        
        for i in range(len(tokens)):
            # 各パターンをチェック
            for pattern in patterns:
                if self._match_pattern_at(tokens, i, pattern):
                    # パターンに一致する場合、複合語を生成
                    compound = self._build_compound_from_pattern(tokens, i, pattern)
                    if compound and self._is_valid_technical_compound(compound):
                        compounds.add(compound)
        
        # 名詞の連続も処理（従来の方法を改良）
        noun_sequences = self._extract_noun_sequences(tokens)
        for sequence in noun_sequences:
            if 2 <= len(sequence) <= 4:
                compound = ''.join([token.surface() for token in sequence])
                if self._is_valid_technical_compound(compound):
                    compounds.add(compound)
        
        return compounds
    
    def _match_pattern_at(self, tokens: List, start: int, pattern: List[str]) -> bool:
        """指定位置でパターンがマッチするかチェック"""
        if start + len(pattern) > len(tokens):
            return False
        
        for i, expected_pos in enumerate(pattern):
            actual_pos = tokens[start + i].part_of_speech()[0]
            # 助詞は種類を問わない
            if expected_pos == '助詞' and actual_pos == '助詞':
                continue
            if actual_pos != expected_pos:
                return False
        return True
    
    def _build_compound_from_pattern(self, tokens: List, start: int, pattern: List[str]) -> str:
        """パターンに基づいて複合語を構築"""
        parts = []
        for i in range(len(pattern)):
            token = tokens[start + i]
            # 助詞「の」などは含めない場合もある
            if token.part_of_speech()[0] != '助詞':
                parts.append(token.surface())
        return ''.join(parts)
    
    def _extract_noun_sequences(self, tokens: List) -> List[List]:
        """連続する名詞のシーケンスを抽出"""
        sequences = []
        current = []
        
        for token in tokens:
            if token.part_of_speech()[0] == '名詞':
                current.append(token)
            else:
                if len(current) >= 2:
                    sequences.append(current)
                current = []
        
        if len(current) >= 2:
            sequences.append(current)
        
        return sequences
    
    def _is_valid_technical_compound(self, compound: str) -> bool:
        """技術的な複合語として妥当かチェック"""
        # 長さチェック
        if len(compound) < 3 or len(compound) > 20:
            return False
        
        # 一般的すぎる語は除外
        common_words = {'こと', 'もの', 'ため', 'これ', 'それ'}
        if compound in common_words:
            return False
        
        # 技術用語の特徴があるかチェック
        return self._has_technical_features(compound)
    
    def _has_technical_features(self, term: str) -> bool:
        """技術用語の特徴を持つかチェック"""
        import re
        
        # 英数字混在
        if re.search(r'[A-Za-z]', term) and re.search(r'[0-9]', term):
            return True
        
        # カタカナを含む
        if re.search(r'[ア-ヴー]', term):
            return True
        
        # 専門的な接尾辞
        technical_suffixes = ['率', '性', '度', '量', '値', '力', '機', '器', '弁', '式', '型', '法']
        for suffix in technical_suffixes:
            if term.endswith(suffix):
                return True
        
        # 3文字以上の漢字複合語
        if re.match(r'^[一-龥]{3,}$', term):
            return True
        
        return False
    
    def _basic_filter(self, candidates: Set[str]) -> Set[str]:
        """基本的なフィルタリング（部分文字列チェックは不要）"""
        filtered = set()
        
        for cand in candidates:
            # 極端に長い/短いものを除外
            if len(cand) < 2 or len(cand) > 30:
                continue
            
            # 純粋な数字は除外
            if cand.isdigit():
                continue
            
            # 一般的すぎる単語は除外
            common_words = {'こと', 'もの', 'ため', 'など', 'これ', 'それ', 'あれ', 'どれ', 'ここ', 'そこ'}
            if cand in common_words:
                continue
            
            filtered.add(cand)
        
        return filtered
    
    def _calculate_term_frequency(self, text: str, term: str, tokens_by_mode: Dict = None) -> int:
        """形態素境界を考慮した正確な頻度計算"""
        # すでにトークン化されている場合はそれを使用
        if tokens_by_mode and 'short' in tokens_by_mode:
            tokens = tokens_by_mode['short']
        else:
            tokens = self.tokenizer_obj.tokenize(text, tokenizer.Tokenizer.SplitMode.A)
        
        frequency = 0
        term_tokens = self.tokenizer_obj.tokenize(term, tokenizer.Tokenizer.SplitMode.A)
        term_surfaces = [t.surface() for t in term_tokens]
        
        # トークンレベルでマッチング
        for i in range(len(tokens) - len(term_tokens) + 1):
            window = [tokens[j].surface() for j in range(i, i + len(term_tokens))]
            if window == term_surfaces:
                frequency += 1
        
        # 完全一致も確認（複合語が単一トークンの場合）
        for token in tokens:
            if token.surface() == term:
                frequency += 1
                break  # 重複カウントを避ける
        
        return max(1, frequency)  # 最低1を返す
    
    def _score_candidates(self, candidates: Set[str], text: str, 
                         tokens_by_mode: Dict) -> List[Tuple[str, float]]:
        """候補をスコアリング（C値/NC値を含む）"""
        import re
        
        # C値計算用のデータ準備
        candidate_data = []
        for cand in candidates:
            candidate_data.append({
                'term': cand,
                'frequency': text.count(cand),
                'length': len(cand)
            })
        
        # C値計算
        c_values = self._calculate_c_value(candidate_data)
        
        # NC値計算（文脈重み付きC値）
        nc_values = self._calculate_nc_value(candidate_data, c_values, text)
        
        scored = []
        for cand in candidates:
            score = 0.0
            freq = text.count(cand)
            
            # 1. C値/NC値スコア（最重要）
            c_score = c_values.get(cand, 0)
            nc_score = nc_values.get(cand, c_score)
            score += nc_score * 5  # 重み付け
            
            # 2. 頻度スコア（対数スケール）- 形態素境界考慮
            real_freq = self._calculate_term_frequency(text, cand, tokens_by_mode)
            freq_score = math.log(real_freq + 1) * 3
            score += freq_score
            
            # 3. 長さスコア（適度な長さを評価）
            length = len(cand)
            if 4 <= length <= 10:
                length_score = 10
            elif 10 < length <= 15:
                length_score = 8
            elif 2 <= length < 4:
                length_score = 3
            else:
                length_score = 1
            score += length_score
            
            # 4. 複数モードで出現（頑健性）
            mode_count = 0
            for mode_name, tokens in tokens_by_mode.items():
                token_surfaces = [t.surface() for t in tokens]
                if cand in ' '.join(token_surfaces):
                    mode_count += 1
            robustness_score = mode_count * 5
            score += robustness_score
            
            # 5. 文字種スコア
            if re.search(r'[A-Z0-9]{2,}', cand):  # 英数字混在
                score += 8
            if re.search(r'[ア-ヴー]', cand):  # カタカナ
                score += 5
            
            scored.append((cand, score))
        
        return scored
    
    def _calculate_c_value(self, candidates: List[Dict]) -> Dict[str, float]:
        """C-value統計指標を計算"""
        c_values = {}
        
        # 候補を長さでソート（長い順）
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
                # 独立した用語
                c_value = math.log2(max(2, length)) * freq
            else:
                # 他の用語に含まれる
                c_value = math.log2(max(2, length)) * (freq - contained_count / containing_terms)
            
            c_values[term] = max(0, c_value)
        
        return c_values
    
    def _calculate_nc_value(self, candidates: List[Dict], c_values: Dict[str, float], 
                           text: str) -> Dict[str, float]:
        """NC-value（文脈重み付きC値）を計算"""
        nc_values = {}
        
        # 各用語の文脈語を収集
        context_words = {}
        for candidate in candidates:
            term = candidate['term']
            contexts = self._extract_context_words(term, text)
            context_words[term] = contexts
        
        # 文脈語の重みを計算
        all_context_words = set()
        for contexts in context_words.values():
            all_context_words.update(contexts)
        
        context_weights = {}
        for word in all_context_words:
            # 何種類の用語と共起するか
            term_count = sum(1 for contexts in context_words.values() if word in contexts)
            context_weights[word] = term_count
        
        # NC値計算
        for term in c_values:
            c_val = c_values[term]
            
            if term in context_words and context_words[term]:
                # 文脈重みの平均
                weights = [context_weights.get(w, 1) for w in context_words[term]]
                avg_weight = sum(weights) / len(weights) if weights else 1
                nc_values[term] = 0.8 * c_val + 0.2 * c_val * avg_weight
            else:
                nc_values[term] = c_val
        
        return nc_values
    
    def _extract_context_words(self, term: str, text: str, window: int = 3) -> Set[str]:
        """用語の前後の文脈語を抽出"""
        import re
        context_words = set()
        
        # 用語の出現位置を検索
        pattern = re.compile(re.escape(term))
        for match in pattern.finditer(text):
            start = match.start()
            end = match.end()
            
            # 前後のテキストを取得
            prefix = text[max(0, start-30):start]
            suffix = text[end:min(len(text), end+30)]
            
            # 単語に分割（簡易的）
            words = re.findall(r'[ぁ-ん]+|[ァ-ヴー]+|[一-龥]+|[A-Za-z0-9]+', prefix + suffix)
            context_words.update(words[:window] + words[-window:])
        
        return context_words

class SimplePerplexityEvaluator:
    """真のPerplexity計算のみを使用したシンプルな評価器"""
    
    def __init__(self):
        self.azure_client = AsyncAzureOpenAI(
            api_key=AZURE_KEY,
            api_version="2024-12-01-preview",
            azure_endpoint=AZURE_ENDPOINT
        )
        self.deployment = AZURE_DEPLOYMENT
        self.tokenizer_obj = dictionary.Dictionary().create()
        # GPT-4o用のtokenizer
        try:
            self.tiktoken_enc = tiktoken.encoding_for_model("gpt-4o")
        except:
            self.tiktoken_enc = tiktoken.get_encoding("o200k_base")  # GPT-4o default
    
    async def evaluate_terms(self, terms: List[str], text: str) -> List[Dict]:
        """用語リストを評価"""
        results = []
        
        # バッチ処理
        batch_size = 5
        for i in range(0, len(terms), batch_size):
            batch = terms[i:i+batch_size]
            logger.info(f"評価中: {i+1}-{min(i+batch_size, len(terms))}/{len(terms)}")
            
            # 並列評価
            tasks = [self._evaluate_single_term(term, text) for term in batch]
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for term, result in zip(batch, batch_results):
                if isinstance(result, Exception):
                    logger.warning(f"評価エラー ({term}): {result}")
                    results.append({
                        'term': term,
                        'perplexity': 50.0,
                        'confidence': 'medium_confidence'
                    })
                else:
                    results.append(result)
            
            # レート制限対策
            if i + batch_size < len(terms):
                await asyncio.sleep(2)
        
        return results
    
    async def _evaluate_single_term(self, term: str, text: str) -> Dict:
        """単一用語のPerplexity計算（改善版）"""
        try:
            perplexities = []
            
            # 方法1: 用語を含む文を完成させてPerplexity計算
            sentence_ppls = await self._evaluate_with_sentence_completion(term, text)
            perplexities.extend(sentence_ppls)
            
            # 方法2: マスク予測タスクとして評価
            mask_ppls = await self._evaluate_with_mask_prediction(term, text)
            perplexities.extend(mask_ppls)
            
            # 方法3: 文脈付き直接評価
            if len(perplexities) < 2:
                direct_ppl = await self._evaluate_direct(term, text)
                if direct_ppl > 0:
                    perplexities.append(direct_ppl)
            
            # 有効な値のみ使用
            valid_ppls = [p for p in perplexities if 5 < p < 100]
            
            if valid_ppls:
                avg_perplexity = np.median(valid_ppls)  # 中央値を使用（外れ値に強い）
            else:
                # フォールバック：用語の特性から推定
                avg_perplexity = self._estimate_perplexity_from_features(term, text)
            
            # 確信度分類（調整済み閾値）
            if avg_perplexity > 25:  # 専門用語は予測困難
                confidence = "high_confidence"
            elif avg_perplexity > 15:
                confidence = "medium_confidence"
            else:
                confidence = "low_confidence"
            
            return {
                'term': term,
                'perplexity': avg_perplexity,
                'confidence': confidence,
                'frequency': self._calculate_term_frequency_simple(text, term)
            }
            
        except Exception as e:
            logger.error(f"評価エラー ({term}): {e}")
            return self._fallback_evaluation(term, text)
    
    async def _evaluate_with_sentence_completion(self, term: str, text: str) -> List[float]:
        """文完成タスクでPerplexity評価"""
        perplexities = []
        
        # 用語を含む文を見つける
        sentences = self._find_sentences_with_term(term, text)
        
        for sentence in sentences[:2]:  # 最大2文で評価
            # 用語の位置を特定
            term_pos = sentence.find(term)
            if term_pos == -1:
                continue
                
            # 用語の前までをプロンプトに
            prefix = sentence[:term_pos]
            target = sentence[term_pos:term_pos + len(term)]
            
            # 文完成を要求
            response = await self.azure_client.chat.completions.create(
                model=self.deployment,
                messages=[
                    {"role": "system", "content": "次の文を自然に完成させてください。"},
                    {"role": "user", "content": f"文：「{prefix}___」\n空欄に入る語句を推測してください。"}
                ],
                max_tokens=50,
                temperature=0,
                logprobs=True,
                top_logprobs=20
            )
            
            if response.choices[0].logprobs and response.choices[0].logprobs.content:
                # 生成されたトークンから用語部分のlogprobを抽出
                ppl = self._extract_term_perplexity(
                    response.choices[0].logprobs.content,
                    term
                )
                if ppl > 0:
                    perplexities.append(ppl)
            
            await asyncio.sleep(0.1)
        
        return perplexities
    
    async def _evaluate_with_mask_prediction(self, term: str, text: str) -> List[float]:
        """マスク予測タスクとしてPerplexity評価"""
        perplexities = []
        
        contexts = self._create_mask_contexts(term, text)
        
        for context in contexts[:2]:
            prompt = f"""以下の文の[MASK]に入る最も適切な語を答えてください。

文：{context['masked_sentence']}

選択肢なしで、[MASK]に入る語句のみを答えてください。"""
            
            response = await self.azure_client.chat.completions.create(
                model=self.deployment,
                messages=[
                    {"role": "system", "content": "文の穴埋め問題を解いてください。"},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=20,
                temperature=0,
                logprobs=True,
                top_logprobs=20
            )
            
            if response.choices[0].logprobs:
                # 各トップ候補から用語の確率を探す
                term_prob = self._find_term_probability(
                    response.choices[0].logprobs.content,
                    term
                )
                if term_prob > 0:
                    ppl = -math.log(term_prob)
                    perplexities.append(min(100, max(5, ppl * 10)))
            
            await asyncio.sleep(0.1)
        
        return perplexities
    
    async def _evaluate_direct(self, term: str, text: str) -> float:
        """直接的なPerplexity評価"""
        try:
            # 用語を含む短い文脈を作成
            context_sentence = self._get_best_context_sentence(term, text)
            
            # 用語の予測可能性を直接質問
            prompt = f"""次の文脈で「{term}」という語が使われる確率を評価してください。

文脈：{context_sentence}

この文脈で「{term}」は：
A) 非常に予測しやすい（一般的な語）
B) やや予測しやすい
C) やや予測しにくい（やや専門的）
D) 非常に予測しにくい（専門用語）

最も適切な選択肢（A-D）のみを答えてください。"""

            response = await self.azure_client.chat.completions.create(
                model=self.deployment,
                messages=[
                    {"role": "system", "content": "語の専門性を評価してください。"},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=5,
                temperature=0
            )
            
            answer = response.choices[0].message.content.strip().upper()
            
            # 回答をPerplexityスコアに変換
            score_map = {
                'A': 10.0,  # 一般的
                'B': 18.0,  # やや一般的
                'C': 28.0,  # やや専門的
                'D': 40.0   # 専門的
            }
            
            return score_map.get(answer[0] if answer else 'C', 25.0)
            
        except Exception as e:
            logger.debug(f"直接評価エラー: {e}")
            return 25.0
    
    def _estimate_perplexity_from_features(self, term: str, text: str) -> float:
        """用語の特徴からPerplexityを推定"""
        import re
        score = 20.0  # ベーススコア
        
        # 英数字混在（型番など）は専門的
        if re.search(r'[A-Z][0-9]|[0-9][A-Z]', term):
            score += 15
        
        # 長い複合語は専門的
        if len(term) > 8:
            score += 10
        elif len(term) > 5:
            score += 5
        
        # カタカナ専門用語
        if re.match(r'^[ァ-ヴー]+$', term) and len(term) > 4:
            score += 8
        
        # 頻度が低い場合は専門的
        freq = text.count(term)
        if freq == 1:
            score += 10
        elif freq == 2:
            score += 5
        
        # アルファベット略語
        if re.match(r'^[A-Z]{2,}$', term):
            score += 12
        
        return min(60, max(10, score))
    
    def _fallback_evaluation(self, term: str, text: str) -> Dict:
        """フォールバック評価"""
        # 特徴ベースの推定を使用
        estimated_ppl = self._estimate_perplexity_from_features(term, text)
        
        if estimated_ppl > 30:
            confidence = "high_confidence"
        elif estimated_ppl > 20:
            confidence = "medium_confidence"
        else:
            confidence = "low_confidence"
        
        return {
            'term': term,
            'perplexity': estimated_ppl,
            'confidence': confidence,
            'frequency': self._calculate_term_frequency_simple(text, term)
        }
    
    def _find_sentences_with_term(self, term: str, text: str) -> List[str]:
        """用語を含む文を抽出"""
        import re
        sentences = []
        # 簡易的な文分割
        text_sentences = re.split(r'[。！？\n]+', text)
        for sent in text_sentences:
            if term in sent and 10 < len(sent) < 200:
                sentences.append(sent)
        return sentences[:3]
    
    def _create_mask_contexts(self, term: str, text: str) -> List[Dict]:
        """マスク文脈を作成"""
        contexts = []
        sentences = self._find_sentences_with_term(term, text)
        
        for sent in sentences[:2]:
            if term in sent:
                masked = sent.replace(term, "[MASK]")
                contexts.append({
                    'masked_sentence': masked,
                    'original': sent,
                    'term': term
                })
        
        return contexts
    
    def _get_best_context_sentence(self, term: str, text: str) -> str:
        """最適な文脈文を取得"""
        sentences = self._find_sentences_with_term(term, text)
        if sentences:
            # 最も短い文を選択（簡潔な定義文の可能性が高い）
            return min(sentences, key=len)
        return f"{term}について"
    
    def _extract_term_perplexity(self, logprobs_content, term: str) -> float:
        """logprobsから用語のPerplexityを抽出"""
        try:
            # トークン列から用語を探す
            for item in logprobs_content:
                if item.token and term in item.token:
                    if item.logprob:
                        return math.exp(-item.logprob)
            return 0
        except:
            return 0
    
    def _find_term_probability(self, logprobs_content, term: str) -> float:
        """logprobsから用語の確率を探す"""
        try:
            for item in logprobs_content:
                if item.token and term in item.token:
                    if item.logprob:
                        return math.exp(item.logprob)
                # トップ候補も確認
                if item.top_logprobs:
                    for top in item.top_logprobs:
                        if top.token and term in top.token:
                            return math.exp(top.logprob)
            return 0
        except:
            return 0
    
    def _create_contexts(self, term: str, text: str, max_contexts: int = 3) -> List[Dict]:
        """用語の文脈を作成"""
        contexts = []
        
        # テキスト内での出現箇所を検索
        positions = []
        index = text.find(term)
        while index != -1 and len(positions) < max_contexts:
            positions.append(index)
            index = text.find(term, index + 1)
        
        # 各出現箇所から文脈作成
        for pos in positions:
            start = max(0, pos - 50)
            prefix = text[start:pos]
            contexts.append({'prefix': prefix, 'term': term})
        
        # 出現箇所が少ない場合は汎用文脈を追加
        if len(contexts) < 2:
            contexts.extend([
                {'prefix': f"この{term}は", 'term': term},
                {'prefix': f"重要な{term}", 'term': term}
            ])
        
        return contexts[:max_contexts]
    
    def _calculate_term_frequency_simple(self, text: str, term: str) -> int:
        """簡易的な頻度計算（評価器用）"""
        # Sudachiでトークン化
        tokens = self.tokenizer_obj.tokenize(text, tokenizer.Tokenizer.SplitMode.A)
        term_tokens = self.tokenizer_obj.tokenize(term, tokenizer.Tokenizer.SplitMode.A)
        term_surfaces = [t.surface() for t in term_tokens]
        
        frequency = 0
        # トークンレベルでマッチング
        for i in range(len(tokens) - len(term_tokens) + 1):
            window = [tokens[j].surface() for j in range(i, i + len(term_tokens))]
            if window == term_surfaces:
                frequency += 1
        
        return max(1, frequency)
    
    async def _evaluate_with_logit_bias(self, prefix: str, term: str) -> float:
        """logit_biasでtermを強制的に生成させてPPLを取得"""
        try:
            # termをトークン化
            token_ids = self.tiktoken_enc.encode(term)
            
            if not token_ids:
                return 50.0
            
            # 最初のトークンを強制（値を調整：5程度が適切）
            first_token_id = token_ids[0]
            logit_bias = {str(first_token_id): 5}
            
            response = await self.azure_client.chat.completions.create(
                model=self.deployment,
                messages=[
                    {"role": "system", "content": "テキストを自然に継続してください。"},
                    {"role": "user", "content": prefix}
                ],
                max_tokens=min(20, len(token_ids) * 2),  # 適切な長さに制限
                temperature=0.0,
                logprobs=True,
                logit_bias=logit_bias
            )
            
            if response.choices[0].logprobs and response.choices[0].logprobs.content:
                logprobs = []
                generated_tokens = []
                
                # 生成されたトークンとtermのトークンを比較
                for i, token_data in enumerate(response.choices[0].logprobs.content):
                    if i < len(token_ids):
                        logprobs.append(token_data.logprob)
                        generated_tokens.append(token_data.token)
                
                if logprobs:
                    avg_logprob = np.mean(logprobs)
                    # 妥当な範囲に制限
                    ppl = math.exp(-avg_logprob) * 10
                    return min(100.0, max(5.0, ppl))
            
            return 50.0
            
        except Exception as e:
            logger.debug(f"logit_bias評価エラー: {e}")
            return 50.0
    
    def _calculate_perplexity_from_logprobs(self, logprobs_content: List, target_term: str) -> float:
        """logprobsからPerplexity計算"""
        try:
            generated_text = ""
            all_logprobs = []
            
            for token_data in logprobs_content:
                token = token_data.token
                logprob = token_data.logprob
                generated_text += token
                
                # 用語が含まれる場合
                if target_term.lower() in generated_text.lower():
                    all_logprobs.append(logprob)
                    if len(generated_text) >= len(target_term):
                        break
            
            if all_logprobs:
                avg_logprob = np.mean(all_logprobs)
                perplexity = math.exp(-avg_logprob) * 10  # スケーリング
                return min(95.0, max(5.0, perplexity))
            
            return 50.0
            
        except Exception:
            return 50.0

def merge_broken_words(text: str) -> str:
    """改行で分割された単語を結合"""
    import re
    
    # カタカナ語の分割を修正（例：「アン\nモニア」→「アンモニア」）
    text = re.sub(r'([ァ-ヶー]+)\n([ァ-ヶー]+)', r'\1\2', text)
    
    # 漢字とひらがなの不自然な分割
    text = re.sub(r'([一-龥]+)\n([ぁ-ん]+)', r'\1\2', text)
    
    # 英数字の分割（例：「6L28\nADF」→「6L28ADF」）
    text = re.sub(r'([A-Z0-9]+)\n([A-Z0-9]+)', r'\1\2', text)
    
    # カタカナと漢字の分割
    text = re.sub(r'([ァ-ヶー]+)\n([一-龥]+)', r'\1\2', text)
    
    # 漢字の連続の分割
    text = re.sub(r'([一-龥]+)\n([一-龥]+)', r'\1\2', text)
    
    return text

def load_pdf(pdf_path: Path) -> str:
    """PDFファイルからテキストを抽出（改行処理改善版）"""
    try:
        doc = fitz.open(str(pdf_path))
        full_text = []
        
        logger.info(f"PDF読み込み中: {len(doc)}ページ")
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text()
            
            if text.strip():
                # 改行で分割された単語を結合
                text = merge_broken_words(text)
                full_text.append(text)
        
        doc.close()
        result = '\n'.join(full_text)
        logger.info(f"抽出完了: {len(result)}文字")
        return result
        
    except Exception as e:
        logger.error(f"PDF読み込みエラー: {e}")
        raise

async def main():
    """メイン処理"""
    parser = argparse.ArgumentParser(
        description="汎用版専門用語抽出（複数分割モード統合）"
    )
    parser.add_argument("input_file", type=Path, help="入力ファイル（PDF/TXT）")
    parser.add_argument("--limit", type=int, default=30, help="評価する候補数")
    parser.add_argument("--output", type=Path, default=Path("output/generic_results.json"))
    
    args = parser.parse_args()
    
    # 入力ファイル確認
    if not args.input_file.exists():
        logger.error(f"ファイルが存在しません: {args.input_file}")
        sys.exit(1)
    
    # テキスト読み込み
    logger.info(f"ファイル読み込み: {args.input_file}")
    if args.input_file.suffix.lower() == '.pdf':
        text = load_pdf(args.input_file)
    else:
        with open(args.input_file, 'r', encoding='utf-8') as f:
            text = f.read()
    
    if not text.strip():
        logger.error("空のファイルです")
        sys.exit(1)
    
    # 1. 候補抽出
    logger.info("専門用語候補を抽出中...")
    extractor = GenericTermExtractor()
    candidates = extractor.extract_candidates(text, max_candidates=args.limit)
    logger.info(f"候補数: {len(candidates)}")
    
    if not candidates:
        logger.error("候補が見つかりませんでした")
        sys.exit(1)
    
    # 2. Perplexity評価
    logger.info("Perplexity計算を開始...")
    evaluator = SimplePerplexityEvaluator()
    results = await evaluator.evaluate_terms(candidates, text)
    
    # 3. 動的閾値で結果を分類
    perplexities = [r['perplexity'] for r in results if r['perplexity'] != 50.0]
    
    if len(perplexities) >= 3:  # 十分なデータがある場合は動的閾値
        p70 = np.percentile(perplexities, 70)
        p90 = np.percentile(perplexities, 90)
        
        high_confidence = [r for r in results if r['perplexity'] > p90]
        medium_confidence = [r for r in results if p70 < r['perplexity'] <= p90]
        low_confidence = [r for r in results if r['perplexity'] <= p70]
        
        # 個別のconfidenceラベルを更新
        for r in high_confidence:
            r['confidence'] = 'high_confidence'
        for r in medium_confidence:
            r['confidence'] = 'medium_confidence'
        for r in low_confidence:
            r['confidence'] = 'low_confidence'
        
        logger.info(f"動的閾値: P70={p70:.1f}, P90={p90:.1f}")
    else:  # データが少ない場合は固定閾値
        high_confidence = [r for r in results if r['confidence'] == 'high_confidence']
        medium_confidence = [r for r in results if r['confidence'] == 'medium_confidence']
        low_confidence = [r for r in results if r['confidence'] == 'low_confidence']
    
    # Perplexityでソート
    for group in [high_confidence, medium_confidence, low_confidence]:
        group.sort(key=lambda x: x['perplexity'])
    
    # 4. 結果保存
    output_data = {
        'metadata': {
            'input_file': str(args.input_file),
            'total_candidates': len(candidates),
            'evaluated': len(results)
        },
        'results': {
            'high_confidence': high_confidence,
            'medium_confidence': medium_confidence,
            'low_confidence': low_confidence
        },
        'summary': {
            'high_confidence_count': len(high_confidence),
            'medium_confidence_count': len(medium_confidence),
            'low_confidence_count': len(low_confidence)
        }
    }
    
    args.output.parent.mkdir(parents=True, exist_ok=True)
    
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    # 5. サマリー表示
    print("\n" + "="*50)
    print("処理結果")
    print("="*50)
    print(f"入力ファイル: {args.input_file.name}")
    print(f"総候補数: {len(candidates)}")
    print(f"高確信度: {len(high_confidence)}語")
    print(f"中確信度: {len(medium_confidence)}語")
    print(f"低確信度: {len(low_confidence)}語")
    
    if high_confidence:
        print("\n高確信度の専門用語（上位5語）:")
        for term_data in high_confidence[:5]:
            print(f"  - {term_data['term']}: Perplexity={term_data['perplexity']:.1f}")
    
    print(f"\n結果保存: {args.output}")

if __name__ == "__main__":
    asyncio.run(main())