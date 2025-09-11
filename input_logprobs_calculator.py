#!/usr/bin/env python3
"""
入力側logprobsを擬似的に計算する改善版Perplexity評価
"""

import asyncio
import logging
import math
from typing import Dict, List, Optional
from openai import AsyncAzureOpenAI
import numpy as np
import os
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

class InputLogprobsCalculator:
    """入力テキストのlogprobsを擬似的に取得"""
    
    def __init__(self):
        self.client = AsyncAzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY") or os.getenv("AZURE_OPENAI_KEY"),
            api_version="2024-12-01-preview",
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
        )
        self.deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o")
    
    async def calculate_term_perplexity(self, text: str, term: str) -> Dict:
        """
        用語の入力Perplexityを擬似的に計算
        
        手法：
        1. 用語の直前までをプロンプトとして与える
        2. モデルに続きを生成させる
        3. 生成部分が用語と一致するか、そのlogprobsを確認
        """
        try:
            # テキスト内の用語位置を全て検索
            positions = []
            idx = text.find(term)
            while idx != -1:
                positions.append(idx)
                idx = text.find(term, idx + 1)
            
            if not positions:
                return {
                    'term': term,
                    'perplexity': 100.0,
                    'confidence': 'not_found',
                    'method': 'input_logprobs'
                }
            
            all_perplexities = []
            
            for pos in positions[:3]:  # 最大3箇所で評価
                # 用語の直前までのテキスト
                prefix = text[:pos]
                # 用語の直後のテキスト（検証用）
                actual_continuation = text[pos:pos + len(term)]
                
                if len(prefix) < 10:  # プレフィックスが短すぎる場合はスキップ
                    continue
                
                # ステップ1: 直接生成
                perplexity = await self._evaluate_direct_generation(
                    prefix, term, actual_continuation
                )
                if perplexity:
                    all_perplexities.append(perplexity)
                
                # ステップ2: 文字単位生成（より細かい評価）
                char_perplexity = await self._evaluate_character_generation(
                    prefix, term
                )
                if char_perplexity:
                    all_perplexities.append(char_perplexity)
            
            if all_perplexities:
                avg_perplexity = np.mean(all_perplexities)
                
                # 確信度分類
                if avg_perplexity < 20:
                    confidence = "high"
                elif avg_perplexity < 50:
                    confidence = "medium"
                else:
                    confidence = "low"
                
                return {
                    'term': term,
                    'perplexity': avg_perplexity,
                    'confidence': confidence,
                    'method': 'input_logprobs',
                    'samples': len(all_perplexities)
                }
            
            return {
                'term': term,
                'perplexity': 50.0,
                'confidence': 'medium',
                'method': 'input_logprobs'
            }
            
        except Exception as e:
            logger.error(f"Perplexity計算エラー: {e}")
            return {
                'term': term,
                'perplexity': 50.0,
                'confidence': 'error',
                'method': 'input_logprobs'
            }
    
    async def _evaluate_direct_generation(self, prefix: str, term: str, 
                                         actual: str) -> Optional[float]:
        """直接生成による評価"""
        try:
            # 最後の50文字程度をコンテキストとして使用
            context = prefix[-200:] if len(prefix) > 200 else prefix
            
            response = await self.client.chat.completions.create(
                model=self.deployment,
                messages=[
                    {"role": "system", "content": "テキストを自然に継続してください。"},
                    {"role": "user", "content": context}
                ],
                max_tokens=len(term) + 5,  # 用語の長さ+余裕
                temperature=0.0,
                logprobs=True,
                top_logprobs=10
            )
            
            if response.choices[0].logprobs and response.choices[0].logprobs.content:
                generated = response.choices[0].message.content
                
                # 生成されたテキストが用語を含むか確認
                if term in generated or generated in term:
                    # 関連するトークンのlogprobsを収集
                    logprobs = []
                    for token_data in response.choices[0].logprobs.content:
                        if any(char in token_data.token for char in term):
                            logprobs.append(token_data.logprob)
                    
                    if logprobs:
                        avg_logprob = np.mean(logprobs)
                        return math.exp(-avg_logprob)
                
                # トップ候補に用語があるか確認
                for token_data in response.choices[0].logprobs.content:
                    if hasattr(token_data, 'top_logprobs'):
                        for candidate in token_data.top_logprobs:
                            if term in candidate.token or candidate.token in term:
                                return math.exp(-candidate.logprob)
            
            return None
            
        except Exception as e:
            logger.debug(f"直接生成エラー: {e}")
            return None
    
    async def _evaluate_character_generation(self, prefix: str, term: str) -> Optional[float]:
        """文字/トークン単位の生成による評価"""
        try:
            context = prefix[-200:] if len(prefix) > 200 else prefix
            accumulated_logprobs = []
            generated = ""
            
            # 1トークンずつ生成
            for i in range(min(5, len(term))):  # 最大5トークン
                prompt = context + generated
                
                response = await self.client.chat.completions.create(
                    model=self.deployment,
                    messages=[
                        {"role": "system", "content": "次の1文字を予測してください。"},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=1,
                    temperature=0.0,
                    logprobs=True,
                    top_logprobs=20
                )
                
                if response.choices[0].logprobs and response.choices[0].logprobs.content:
                    token_data = response.choices[0].logprobs.content[0]
                    generated_token = token_data.token
                    
                    # 正解文字との比較
                    if i < len(term):
                        target_char = term[i:i+len(generated_token)]
                        
                        if generated_token == target_char:
                            # 正解した場合のlogprob
                            accumulated_logprobs.append(token_data.logprob)
                        else:
                            # トップ候補から探す
                            found = False
                            if hasattr(token_data, 'top_logprobs'):
                                for candidate in token_data.top_logprobs:
                                    if candidate.token == target_char:
                                        accumulated_logprobs.append(candidate.logprob)
                                        found = True
                                        break
                            
                            if not found:
                                # 見つからない場合は低い確率を仮定
                                accumulated_logprobs.append(-5.0)
                    
                    generated += generated_token
                    
                    # 用語が完成したら終了
                    if term in generated:
                        break
            
            if accumulated_logprobs:
                avg_logprob = np.mean(accumulated_logprobs)
                return math.exp(-avg_logprob)
            
            return None
            
        except Exception as e:
            logger.debug(f"文字生成エラー: {e}")
            return None
    
    async def batch_evaluate(self, text: str, terms: List[str]) -> List[Dict]:
        """複数用語のバッチ評価"""
        results = []
        
        for term in terms:
            result = await self.calculate_term_perplexity(text, term)
            results.append(result)
            await asyncio.sleep(0.5)  # レート制限対策
        
        return results


# テスト関数
async def test_input_logprobs():
    """入力logprobs計算のテスト"""
    calculator = InputLogprobsCalculator()
    
    text = """
    理論空燃比は内燃機関において重要な指標である。
    アンモニア燃料の理論空燃比を正確に計算することが必要だ。
    エンジンの効率を最適化するために、理論空燃比の理解は不可欠である。
    """
    
    terms = ["理論空燃比", "アンモニア", "エンジン", "内燃機関"]
    
    print("=== 入力側Perplexity評価 ===\n")
    
    for term in terms:
        result = await calculator.calculate_term_perplexity(text, term)
        print(f"用語: {term}")
        print(f"  Perplexity: {result['perplexity']:.2f}")
        print(f"  確信度: {result['confidence']}")
        print(f"  手法: {result['method']}")
        print()

if __name__ == "__main__":
    asyncio.run(test_input_logprobs())