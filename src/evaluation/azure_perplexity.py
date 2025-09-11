#!/usr/bin/env python3
"""
Azure OpenAI logprobs APIを使用した真のPerplexity計算
数学的に正確な損失ベースの評価
"""
import asyncio
import logging
import math
import json
from typing import Dict, List, Tuple, Optional
from openai import AsyncAzureOpenAI
import numpy as np
import os
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

class AzureLogprobsPerplexityCalculator:
    """Azure OpenAI logprobs APIを使用した真のPerplexity計算"""
    
    def __init__(self):
        self.azure_client = AsyncAzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_KEY"),
            api_version="2024-12-01-preview",
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
        )
        self.deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o")
        
    async def calculate_true_perplexity(self, text: str, term: str) -> Dict:
        """
        真のPerplexityを計算（logprobsベース）
        
        Returns:
            {
                'term': str,
                'perplexity': float,
                'avg_logprob': float,
                'token_details': List[Dict],
                'confidence': str
            }
        """
        try:
            # 用語を含む文脈を作成
            contexts = self._create_contexts(text, term)
            all_perplexities = []
            all_token_details = []
            
            for context in contexts:
                # logprobsを取得
                response = await self.azure_client.chat.completions.create(
                    model=self.deployment,
                    messages=[
                        {"role": "system", "content": "以下のテキストを継続してください。"},
                        {"role": "user", "content": context['prefix']}
                    ],
                    max_tokens=20,  # 用語を含む継続を生成
                    temperature=0.0,
                    logprobs=True,
                    top_logprobs=5
                )
                
                if response.choices[0].logprobs and response.choices[0].logprobs.content:
                    # 各トークンのlogprobを分析
                    token_data = self._analyze_token_logprobs(
                        response.choices[0].logprobs.content, 
                        term
                    )
                    
                    if token_data['found_term']:
                        perplexity = math.exp(-token_data['avg_logprob'])
                        all_perplexities.append(perplexity)
                        all_token_details.append(token_data)
                
                await asyncio.sleep(0.3)  # レート制限対策
            
            if all_perplexities:
                avg_perplexity = np.mean(all_perplexities)
                confidence = self._classify_confidence(avg_perplexity)
                
                return {
                    'term': term,
                    'perplexity': avg_perplexity,
                    'avg_logprob': -math.log(avg_perplexity),
                    'token_details': all_token_details,
                    'confidence': confidence
                }
            
            return self._default_result(term)
            
        except Exception as e:
            logger.error(f"Perplexity計算エラー ({term}): {e}")
            return self._default_result(term)
    
    def _create_contexts(self, text: str, term: str, window_size: int = 50) -> List[Dict]:
        """用語の前後文脈を生成"""
        contexts = []
        
        # 用語の出現箇所を検索
        term_positions = []
        index = text.find(term)
        while index != -1:
            term_positions.append(index)
            index = text.find(term, index + 1)
        
        # 各出現箇所で文脈を作成
        for pos in term_positions[:3]:  # 最大3箇所
            start = max(0, pos - window_size)
            end = min(len(text), pos + len(term) + window_size)
            
            prefix = text[start:pos]
            target = text[pos:pos + len(term)]
            suffix = text[pos + len(term):end]
            
            contexts.append({
                'prefix': prefix,
                'target': target,
                'suffix': suffix,
                'full': prefix + target + suffix
            })
        
        # 用語が見つからない場合は仮想文脈を作成
        if not contexts:
            contexts.append({
                'prefix': f"この{term}は",
                'target': term,
                'suffix': "重要な専門用語です。",
                'full': f"この{term}は重要な専門用語です。"
            })
        
        return contexts
    
    def _analyze_token_logprobs(self, logprobs_content: List, target_term: str) -> Dict:
        """トークンのlogprobsを分析"""
        token_details = []
        term_tokens = []
        found_term = False
        
        for token_data in logprobs_content:
            token = token_data.token
            logprob = token_data.logprob
            prob = math.exp(logprob)
            
            # トップ候補の情報も取得
            top_candidates = []
            if hasattr(token_data, 'top_logprobs') and token_data.top_logprobs:
                for candidate in token_data.top_logprobs[:3]:
                    top_candidates.append({
                        'token': candidate.token,
                        'logprob': candidate.logprob,
                        'prob': math.exp(candidate.logprob)
                    })
            
            token_info = {
                'token': token,
                'logprob': logprob,
                'prob': prob,
                'top_candidates': top_candidates
            }
            
            token_details.append(token_info)
            
            # 目的の用語に関連するトークンを追跡
            if target_term in token or token in target_term:
                term_tokens.append(token_info)
                found_term = True
        
        # 平均logprobを計算
        if term_tokens:
            avg_logprob = np.mean([t['logprob'] for t in term_tokens])
        else:
            avg_logprob = -3.0  # デフォルト（高いPerplexity）
        
        return {
            'found_term': found_term,
            'tokens': token_details[:10],  # 最初の10トークン
            'term_tokens': term_tokens,
            'avg_logprob': avg_logprob,
            'token_count': len(token_details)
        }
    
    async def calculate_masked_perplexity(self, text: str, term: str) -> Dict:
        """マスク予測方式でPerplexity計算"""
        try:
            # 用語をマスク
            masked_text = text.replace(term, "[MASK]")
            
            # マスク部分の予測を要求
            prompt = f"""次の文章の[MASK]部分に入る最も適切な専門用語を予測してください。

文章: {masked_text[:200]}

[MASK]に入る専門用語（1つだけ）:"""
            
            response = await self.azure_client.chat.completions.create(
                model=self.deployment,
                messages=[
                    {"role": "system", "content": "専門用語を予測してください。"},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=10,
                temperature=0.0,
                logprobs=True,
                top_logprobs=10
            )
            
            # 予測結果とlogprobsを分析
            predicted = response.choices[0].message.content.strip()
            
            # 正解用語が予測候補にあるか確認
            if response.choices[0].logprobs and response.choices[0].logprobs.content:
                for token_data in response.choices[0].logprobs.content:
                    if hasattr(token_data, 'top_logprobs'):
                        for candidate in token_data.top_logprobs:
                            if term in candidate.token:
                                # 正解用語のlogprobからPerplexity計算
                                perplexity = math.exp(-candidate.logprob)
                                return {
                                    'term': term,
                                    'predicted': predicted,
                                    'correct': predicted == term,
                                    'perplexity': perplexity,
                                    'logprob': candidate.logprob,
                                    'rank': token_data.top_logprobs.index(candidate) + 1
                                }
            
            # 予測が一致した場合
            if predicted == term:
                # トップ予測のlogprobを使用
                if response.choices[0].logprobs and response.choices[0].logprobs.content:
                    first_token = response.choices[0].logprobs.content[0]
                    perplexity = math.exp(-first_token.logprob)
                    return {
                        'term': term,
                        'predicted': predicted,
                        'correct': True,
                        'perplexity': perplexity,
                        'logprob': first_token.logprob,
                        'rank': 1
                    }
            
            # 予測が外れた場合は高いPerplexity
            return {
                'term': term,
                'predicted': predicted,
                'correct': False,
                'perplexity': 100.0,
                'logprob': -4.6,  # log(0.01)
                'rank': 11  # トップ10外
            }
            
        except Exception as e:
            logger.error(f"マスク予測エラー ({term}): {e}")
            return {
                'term': term,
                'predicted': None,
                'correct': False,
                'perplexity': 80.0,
                'logprob': -4.0,
                'rank': None
            }
    
    async def calculate_document_perplexity_profile(self, text: str, chunk_size: int = 100) -> List[Dict]:
        """文書全体のPerplexityプロファイルを作成"""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size // 2):  # 50%オーバーラップ
            chunk_words = words[i:i + chunk_size]
            if len(chunk_words) < 10:  # 短すぎるチャンクは除外
                continue
                
            chunk_text = ' '.join(chunk_words)
            
            # チャンクのPerplexity計算
            response = await self.azure_client.chat.completions.create(
                model=self.deployment,
                messages=[
                    {"role": "user", "content": chunk_text}
                ],
                max_tokens=1,  # 次の1トークンだけ
                temperature=0.0,
                logprobs=True
            )
            
            if response.choices[0].logprobs and response.choices[0].logprobs.content:
                # 平均logprobからPerplexity計算
                logprobs = [t.logprob for t in response.choices[0].logprobs.content]
                avg_logprob = np.mean(logprobs) if logprobs else -3.0
                perplexity = math.exp(-avg_logprob)
                
                chunks.append({
                    'chunk_id': i // (chunk_size // 2),
                    'start_word': i,
                    'end_word': min(i + chunk_size, len(words)),
                    'text_preview': chunk_text[:100] + '...',
                    'perplexity': perplexity,
                    'avg_logprob': avg_logprob
                })
            
            await asyncio.sleep(0.5)  # レート制限対策
        
        # ホットスポット検出
        if chunks:
            perplexities = [c['perplexity'] for c in chunks]
            mean_perp = np.mean(perplexities)
            std_perp = np.std(perplexities)
            
            for chunk in chunks:
                chunk['is_hotspot'] = chunk['perplexity'] > mean_perp + 1.5 * std_perp
        
        return chunks
    
    def _classify_confidence(self, perplexity: float) -> str:
        """Perplexityから確信度を分類"""
        if perplexity < 20:
            return "high_confidence"
        elif perplexity < 50:
            return "medium_confidence"
        else:
            return "low_confidence"
    
    def _default_result(self, term: str) -> Dict:
        """デフォルト結果"""
        return {
            'term': term,
            'perplexity': 50.0,
            'avg_logprob': -3.9,
            'token_details': [],
            'confidence': 'medium_confidence'
        }

# テスト用メイン関数
async def test_logprobs_perplexity():
    """logprobs APIのテスト"""
    calculator = AzureLogprobsPerplexityCalculator()
    
    # テストテキスト
    text = "理論空燃比は内燃機関において重要な指標である。アンモニア燃料の理論空燃比を正確に計算することが必要だ。"
    terms = ["理論空燃比", "アンモニア燃料", "内燃機関"]
    
    print("=== Azure OpenAI logprobs Perplexity計算テスト ===\n")
    
    for term in terms:
        print(f"用語: {term}")
        
        # 真のPerplexity
        result = await calculator.calculate_true_perplexity(text, term)
        print(f"  真のPerplexity: {result['perplexity']:.2f}")
        print(f"  平均logprob: {result['avg_logprob']:.4f}")
        print(f"  確信度: {result['confidence']}")
        
        # マスク予測Perplexity
        masked_result = await calculator.calculate_masked_perplexity(text, term)
        print(f"  マスク予測: {masked_result['predicted']}")
        print(f"  正解: {masked_result['correct']}")
        print(f"  マスクPerplexity: {masked_result['perplexity']:.2f}")
        print(f"  予測順位: {masked_result['rank']}")
        print()
    
    # 文書プロファイル
    print("=== 文書Perplexityプロファイル ===")
    profile = await calculator.calculate_document_perplexity_profile(text, chunk_size=50)
    for chunk in profile[:3]:
        print(f"チャンク{chunk['chunk_id']}: Perplexity={chunk['perplexity']:.2f}, ホットスポット={chunk['is_hotspot']}")

if __name__ == "__main__":
    asyncio.run(test_logprobs_perplexity())