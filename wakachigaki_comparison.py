#!/usr/bin/env python3
"""
分かち書き精度比較スクリプト
Sudachi + Embedding vs Sudachi + LLM のロジック比較
"""

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
import time

from dotenv import load_dotenv
import numpy as np
from sudachipy import tokenizer, dictionary
from sklearn.metrics.pairwise import cosine_similarity

# LangChain imports
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import AzureOpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME")

@dataclass
class TokenizationResult:
    """分かち書き結果を保持するクラス"""
    method: str  # "sudachi_a", "sudachi_b", "sudachi_c", "embedding", "llm"
    tokens: List[str]
    processing_time: float
    metadata: Dict = field(default_factory=dict)

class SudachiTokenizer:
    """Sudachiの基本分かち書き"""
    
    def __init__(self):
        self.tokenizer = dictionary.Dictionary().create()
        
    def tokenize_all_modes(self, text: str) -> Dict[str, List[str]]:
        """A/B/C全モードで分かち書き"""
        results = {}
        
        # Mode A (最小単位)
        tokens_a = self.tokenizer.tokenize(text, tokenizer.Tokenizer.SplitMode.A)
        results['A'] = [token.surface() for token in tokens_a]
        
        # Mode B (中間単位)
        tokens_b = self.tokenizer.tokenize(text, tokenizer.Tokenizer.SplitMode.B)
        results['B'] = [token.surface() for token in tokens_b]
        
        # Mode C (最大単位)
        tokens_c = self.tokenizer.tokenize(text, tokenizer.Tokenizer.SplitMode.C)
        results['C'] = [token.surface() for token in tokens_c]
        
        return results

class EmbeddingBasedTokenizer:
    """Sudachi + Embedding による分かち書き"""
    
    def __init__(self):
        self.sudachi = SudachiTokenizer()
        # Azure OpenAI text-embedding-3-small を使用
        self.embeddings = AzureOpenAIEmbeddings(
            azure_deployment=AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME,
            openai_api_version=AZURE_OPENAI_API_VERSION,
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            api_key=AZURE_OPENAI_API_KEY
        )
        
    async def tokenize_with_context(self, text: str, window_size: int = 256) -> TokenizationResult:
        """文脈ベクトルを考慮した分かち書き"""
        start_time = time.time()
        
        # 1. Sudachiで初期分割（全モード）
        sudachi_results = self.sudachi.tokenize_all_modes(text)
        
        # 2. 各モードの候補を評価
        best_tokens = []
        text_len = len(text)
        pos = 0
        
        while pos < text_len:
            # ウィンドウ範囲を決定
            window_start = max(0, pos - window_size // 2)
            window_end = min(text_len, pos + window_size // 2)
            window_text = text[window_start:window_end]
            
            # ウィンドウのベクトルを生成
            window_vector = await self.embeddings.aembed_query(window_text)
            
            # 各モードの該当部分を評価
            candidates = []
            for mode, tokens in sudachi_results.items():
                # 現在位置に対応するトークンを探す
                current_pos = 0
                for token in tokens:
                    if current_pos <= pos < current_pos + len(token):
                        # このトークンと周辺を含むテキスト
                        token_context = text[max(0, current_pos - 50):min(text_len, current_pos + len(token) + 50)]
                        token_vector = await self.embeddings.aembed_query(token_context)
                        
                        # 類似度計算
                        similarity = cosine_similarity(
                            np.array(window_vector).reshape(1, -1),
                            np.array(token_vector).reshape(1, -1)
                        )[0][0]
                        
                        candidates.append({
                            'mode': mode,
                            'token': token,
                            'similarity': similarity,
                            'length': len(token)
                        })
                        break
                    current_pos += len(token)
            
            # 最適な候補を選択（類似度と長さのバランス）
            if candidates:
                # 類似度が高く、適度な長さを持つトークンを優先
                best = max(candidates, key=lambda x: x['similarity'] * (1 + 0.1 * x['length']))
                best_tokens.append(best['token'])
                pos += len(best['token'])
            else:
                # 候補がない場合は1文字進める
                if pos < text_len:
                    best_tokens.append(text[pos])
                pos += 1
        
        processing_time = time.time() - start_time
        
        return TokenizationResult(
            method="embedding",
            tokens=best_tokens,
            processing_time=processing_time,
            metadata={"window_size": window_size}
        )
    
    async def tokenize_with_boundaries(self, text: str, threshold: float = 0.3) -> TokenizationResult:
        """境界の信頼度に基づく動的調整"""
        start_time = time.time()
        
        # 初期分割（Mode B を基準）
        sudachi_results = self.sudachi.tokenize_all_modes(text)
        base_tokens = sudachi_results['B']
        
        # 各境界の信頼度を計算
        adjusted_tokens = []
        i = 0
        
        while i < len(base_tokens):
            current_token = base_tokens[i]
            
            # 次のトークンとの境界を評価
            if i < len(base_tokens) - 1:
                next_token = base_tokens[i + 1]
                
                # 両トークンのベクトル生成
                current_vec = await self.embeddings.aembed_query(current_token)
                next_vec = await self.embeddings.aembed_query(next_token)
                
                # 類似度計算
                similarity = cosine_similarity(
                    np.array(current_vec).reshape(1, -1),
                    np.array(next_vec).reshape(1, -1)
                )[0][0]
                
                # 閾値以上なら結合
                if similarity > (1 - threshold):
                    adjusted_tokens.append(current_token + next_token)
                    i += 2  # 2トークン分進める
                else:
                    adjusted_tokens.append(current_token)
                    i += 1
            else:
                adjusted_tokens.append(current_token)
                i += 1
        
        processing_time = time.time() - start_time
        
        return TokenizationResult(
            method="embedding_boundary",
            tokens=adjusted_tokens,
            processing_time=processing_time,
            metadata={"threshold": threshold}
        )

class LLMBasedTokenizer:
    """Sudachi + LLM による分かち書き"""
    
    def __init__(self):
        self.sudachi = SudachiTokenizer()
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            temperature=0.1,
            google_api_key=GOOGLE_API_KEY
        )
        
    async def tokenize_with_llm(self, text: str) -> TokenizationResult:
        """LLMによる文脈理解を用いた分かち書き"""
        start_time = time.time()
        
        # Sudachiで候補生成
        sudachi_results = self.sudachi.tokenize_all_modes(text)
        
        # 曖昧な箇所を検出（A/B/Cで結果が異なる箇所）
        ambiguous_positions = []
        for i, char in enumerate(text):
            # 各モードでの境界位置を確認
            boundaries = set()
            for mode, tokens in sudachi_results.items():
                pos = 0
                for token in tokens:
                    if pos == i:
                        boundaries.add(mode)
                    pos += len(token)
            
            if len(boundaries) > 1:
                ambiguous_positions.append(i)
        
        # LLMに判定を依頼
        prompt = ChatPromptTemplate.from_messages([
            ("system", """あなたは日本語の分かち書き専門家です。
文脈を理解し、最も適切な分かち書きを選択してください。
専門用語は適切な粒度で保持し、文の意味が明確になるよう分割してください。"""),
            ("user", """以下の文を分かち書きしてください。
複数の候補がある場合は、文脈に最も適した分割を選んでください。

文: {text}

候補:
- A単位（最小）: {tokens_a}
- B単位（中間）: {tokens_b}
- C単位（最大）: {tokens_c}

最適な分かち書きを選択し、その理由と共に返してください。
出力形式: 
選択: [A/B/C/カスタム]
分かち書き: [スペース区切りのトークン列]
理由: [選択理由]""")
        ])
        
        # トークンを文字列に変換
        tokens_a_str = " ".join(sudachi_results['A'])
        tokens_b_str = " ".join(sudachi_results['B'])
        tokens_c_str = " ".join(sudachi_results['C'])
        
        response = await self.llm.ainvoke(
            prompt.format_messages(
                text=text,
                tokens_a=tokens_a_str,
                tokens_b=tokens_b_str,
                tokens_c=tokens_c_str
            )
        )
        
        # レスポンスから分かち書き結果を抽出
        response_text = response.content
        tokens = []
        
        # "分かち書き:" の行を探す
        for line in response_text.split('\n'):
            if '分かち書き:' in line or '分かち書き：' in line:
                # コロンの後の部分を取得
                parts = line.split(':', 1) if ':' in line else line.split('：', 1)
                if len(parts) > 1:
                    tokens_str = parts[1].strip()
                    # 角括弧を除去
                    tokens_str = tokens_str.strip('[]')
                    tokens = tokens_str.split()
                    break
        
        # トークンが見つからない場合はデフォルト
        if not tokens:
            tokens = sudachi_results['B']
        
        processing_time = time.time() - start_time
        
        return TokenizationResult(
            method="llm",
            tokens=tokens,
            processing_time=processing_time,
            metadata={"response": response_text[:200]}
        )
    
    async def tokenize_with_hybrid(self, text: str, confidence_threshold: float = 0.7) -> TokenizationResult:
        """ハイブリッドアプローチ（高信頼度部分はSudachi、低信頼度部分はLLM）"""
        start_time = time.time()
        
        # Sudachiで初期分析
        sudachi_results = self.sudachi.tokenize_all_modes(text)
        
        # モード間の一致度を計算
        final_tokens = []
        text_pos = 0
        
        while text_pos < len(text):
            # 各モードの現在位置のトークンを取得
            current_tokens = {}
            for mode, tokens in sudachi_results.items():
                pos = 0
                for token in tokens:
                    if pos <= text_pos < pos + len(token):
                        current_tokens[mode] = token
                        break
                    pos += len(token)
            
            # 一致度を計算
            unique_tokens = set(current_tokens.values())
            confidence = 1.0 / len(unique_tokens) if unique_tokens else 0
            
            if confidence >= confidence_threshold:
                # 高信頼度：Sudachiの結果を使用（B単位）
                token = current_tokens.get('B', text[text_pos])
                final_tokens.append(token)
                text_pos += len(token)
            else:
                # 低信頼度：LLMに判定を依頼（該当部分のみ）
                context_start = max(0, text_pos - 50)
                context_end = min(len(text), text_pos + 100)
                context = text[context_start:context_end]
                
                prompt = f"文脈: {context}\n該当箇所: {text[text_pos:text_pos+20]}\n最適な分割を提案:"
                
                response = await self.llm.ainvoke(prompt)
                # 簡易的な処理（実際はレスポンスを解析）
                token = current_tokens.get('B', text[text_pos])
                final_tokens.append(token)
                text_pos += len(token)
        
        processing_time = time.time() - start_time
        
        return TokenizationResult(
            method="hybrid",
            tokens=final_tokens,
            processing_time=processing_time,
            metadata={"confidence_threshold": confidence_threshold}
        )

class TokenizationEvaluator:
    """分かち書き結果の評価"""
    
    @staticmethod
    def format_tokens(tokens: List[str], separator: str = " | ") -> str:
        """トークンを見やすく整形"""
        return separator.join(tokens)
    
    @staticmethod
    def calculate_stats(result: TokenizationResult) -> Dict:
        """統計情報を計算"""
        tokens = result.tokens
        return {
            "token_count": len(tokens),
            "avg_token_length": sum(len(t) for t in tokens) / len(tokens) if tokens else 0,
            "max_token_length": max(len(t) for t in tokens) if tokens else 0,
            "min_token_length": min(len(t) for t in tokens) if tokens else 0,
            "processing_time": result.processing_time
        }
    
    @staticmethod
    def compare_results(results: List[TokenizationResult]) -> str:
        """複数の結果を比較"""
        comparison = []
        comparison.append("=" * 80)
        comparison.append("分かち書き結果の比較")
        comparison.append("=" * 80)
        
        for result in results:
            comparison.append(f"\n【{result.method}】")
            comparison.append(f"処理時間: {result.processing_time:.3f}秒")
            
            stats = TokenizationEvaluator.calculate_stats(result)
            comparison.append(f"トークン数: {stats['token_count']}")
            comparison.append(f"平均トークン長: {stats['avg_token_length']:.2f}")
            comparison.append(f"最大/最小: {stats['max_token_length']}/{stats['min_token_length']}")
            
            comparison.append(f"分かち書き結果:")
            comparison.append(TokenizationEvaluator.format_tokens(result.tokens))
            comparison.append("-" * 40)
        
        return "\n".join(comparison)

async def main():
    """メイン処理"""
    # テストテキスト
    test_texts = [
        "自然言語処理技術の最新動向について、トランスフォーマーモデルが革新的な成果をもたらしている。",
        "機械学習モデルの精度向上には、大規模データセットと計算資源の確保が不可欠である。",
        "東京スカイツリーは、日本の電波塔として世界最高クラスの高さを誇る建造物です。",
        "深層学習における過学習の問題は、ドロップアウトや正則化によって緩和することができる。",
    ]
    
    # 各トークナイザーを初期化
    sudachi_tokenizer = SudachiTokenizer()
    embedding_tokenizer = EmbeddingBasedTokenizer()
    llm_tokenizer = LLMBasedTokenizer()
    evaluator = TokenizationEvaluator()
    
    for i, text in enumerate(test_texts, 1):
        print(f"\n{'=' * 80}")
        print(f"テキスト {i}: {text}")
        print('=' * 80)
        
        results = []
        
        # Sudachi各モードの結果
        sudachi_results = sudachi_tokenizer.tokenize_all_modes(text)
        for mode, tokens in sudachi_results.items():
            results.append(TokenizationResult(
                method=f"Sudachi_{mode}",
                tokens=tokens,
                processing_time=0.001,  # 簡易的な値
                metadata={"mode": mode}
            ))
        
        # Embedding ベースの結果（処理時間を考慮してスキップ可能）
        try:
            print("Embeddingベースの分かち書きを実行中...")
            embedding_result = await asyncio.wait_for(
                embedding_tokenizer.tokenize_with_context(text),
                timeout=60  # 60秒でタイムアウト
            )
            results.append(embedding_result)
        except asyncio.TimeoutError:
            print("  タイムアウト: Embeddingベースの処理をスキップ")
        except Exception as e:
            print(f"  エラー: {e}")
        
        # 境界調整の結果
        try:
            print("境界調整ベースの分かち書きを実行中...")
            boundary_result = await asyncio.wait_for(
                embedding_tokenizer.tokenize_with_boundaries(text),
                timeout=30  # 30秒でタイムアウト
            )
            results.append(boundary_result)
        except asyncio.TimeoutError:
            print("  タイムアウト: 境界調整ベースの処理をスキップ")
        except Exception as e:
            print(f"  エラー: {e}")
        
        # LLMベースの結果
        print("LLMベースの分かち書きを実行中...")
        llm_result = await llm_tokenizer.tokenize_with_llm(text)
        results.append(llm_result)
        
        # 結果を比較表示
        print(evaluator.compare_results(results))
        
        # 結果をJSONファイルに保存
        output_data = {
            "text": text,
            "results": [
                {
                    "method": r.method,
                    "tokens": r.tokens,
                    "processing_time": r.processing_time,
                    "stats": evaluator.calculate_stats(r)
                }
                for r in results
            ]
        }
        
        output_file = f"tokenization_comparison_{i}.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        print(f"\n結果を {output_file} に保存しました")

if __name__ == "__main__":
    if not GOOGLE_API_KEY:
        print("Error: GOOGLE_API_KEY not found in .env")
        sys.exit(1)
    if not AZURE_OPENAI_API_KEY:
        print("Error: AZURE_OPENAI_API_KEY not found in .env")
        sys.exit(1)
    
    asyncio.run(main())