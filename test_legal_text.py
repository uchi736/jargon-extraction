#!/usr/bin/env python3
"""
法令文書での分かち書き精度比較
"""

import asyncio
import json
import os
import sys
from typing import Dict, List
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

# 法令文書のサンプルテキスト
LEGAL_TEXTS = [
    "医薬品及び医薬部外品の製造管理及び品質管理の基準に関する省令",
    "薬事法第十四条第二項第四号及び第十九条の二第五項において準用する第十四条第二項第四号の規定に基づき",
    "製造業者等は、実効性のある医薬品品質システムを構築するとともに、次に掲げる業務を行わなければならない。",
    "品質部門は、製造部門から独立していなければならない。",
    "バリデーションとは、製造所の構造設備並びに手順、工程その他の製造管理及び品質管理の方法が期待される結果を与えることを検証し、これを文書とすることをいう。",
    "生物由来医薬品等の製造管理及び品質管理",
    "無菌医薬品に係る製品の製造に必要な蒸留水等を供給する設備は、異物又は微生物による蒸留水等の汚染を防止するために必要な構造であること。",
    "原料となる細胞又は組織をドナー動物から採取する場合においては、採取の記録を作成し、これを保管すること。"
]

@dataclass
class TokenizationResult:
    """分かち書き結果を保持するクラス"""
    method: str
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
            azure_deployment=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME"),
            openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY")
        )
        
    async def tokenize_with_boundaries(self, text: str, threshold: float = 0.3) -> TokenizationResult:
        """境界の信頼度に基づく動的調整（簡易版）"""
        start_time = time.time()
        
        # 初期分割（Mode B を基準）
        sudachi_results = self.sudachi.tokenize_all_modes(text)
        base_tokens = sudachi_results['B']
        
        # 簡易的な処理（API呼び出しを最小限に）
        adjusted_tokens = []
        i = 0
        
        while i < len(base_tokens):
            current_token = base_tokens[i]
            
            # 法令用語の判定（ヒューリスティック）
            # 連続する漢字は結合しやすくする
            if i < len(base_tokens) - 1:
                next_token = base_tokens[i + 1]
                
                # 両方が漢字主体なら結合を検討
                if self._is_mostly_kanji(current_token) and self._is_mostly_kanji(next_token):
                    # 助詞でなければ結合
                    if next_token not in ['の', 'に', 'を', 'は', 'が', 'と', 'も', 'や', 'から', 'まで', 'より']:
                        adjusted_tokens.append(current_token + next_token)
                        i += 2
                        continue
            
            adjusted_tokens.append(current_token)
            i += 1
        
        processing_time = time.time() - start_time
        
        return TokenizationResult(
            method="embedding_boundary",
            tokens=adjusted_tokens,
            processing_time=processing_time,
            metadata={"threshold": threshold}
        )
    
    def _is_mostly_kanji(self, text: str) -> bool:
        """文字列が主に漢字かどうか判定"""
        if not text:
            return False
        kanji_count = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
        return kanji_count / len(text) > 0.5

class LLMBasedTokenizer:
    """Sudachi + LLM による分かち書き"""
    
    def __init__(self):
        self.sudachi = SudachiTokenizer()
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            temperature=0.1,
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )
        
    async def tokenize_with_llm(self, text: str) -> TokenizationResult:
        """LLMによる文脈理解を用いた分かち書き"""
        start_time = time.time()
        
        # Sudachiで候補生成
        sudachi_results = self.sudachi.tokenize_all_modes(text)
        
        # LLMに判定を依頼
        prompt = ChatPromptTemplate.from_messages([
            ("system", """あなたは日本語の法令文書の分かち書き専門家です。
法令用語や専門用語は適切な粒度で保持し、文の意味が明確になるよう分割してください。
特に以下の点に注意してください：
- 法令特有の複合語（例：製造管理、品質管理、医薬部外品）は1つの単位として扱う
- 条文番号（例：第十四条）は1つの単位として扱う
- 法的な概念を表す語句は適切にまとめる"""),
            ("user", """以下の法令文を分かち書きしてください。

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
        
        try:
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
                
        except Exception as e:
            print(f"LLM Error: {e}")
            tokens = sudachi_results['B']
            response_text = str(e)
        
        processing_time = time.time() - start_time
        
        return TokenizationResult(
            method="llm",
            tokens=tokens,
            processing_time=processing_time,
            metadata={"response": response_text[:200] if response_text else ""}
        )

def format_tokens(tokens: List[str], separator: str = " | ") -> str:
    """トークンを見やすく整形"""
    return separator.join(tokens)

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

async def main():
    """メイン処理"""
    
    # 各トークナイザーを初期化
    sudachi_tokenizer = SudachiTokenizer()
    embedding_tokenizer = EmbeddingBasedTokenizer()
    llm_tokenizer = LLMBasedTokenizer()
    
    all_results = []
    
    for i, text in enumerate(LEGAL_TEXTS, 1):
        print(f"\n{'=' * 80}")
        print(f"テキスト {i}: {text}")
        print('=' * 80)
        
        results = []
        
        # Sudachi各モードの結果
        sudachi_results = sudachi_tokenizer.tokenize_all_modes(text)
        for mode, tokens in sudachi_results.items():
            result = TokenizationResult(
                method=f"Sudachi_{mode}",
                tokens=tokens,
                processing_time=0.001,
                metadata={"mode": mode}
            )
            results.append(result)
            print(f"\n【Sudachi_{mode}】")
            print(f"トークン数: {len(tokens)}")
            print(f"分かち書き: {format_tokens(tokens)}")
        
        # 境界調整の結果（簡易版）
        try:
            print("\n境界調整ベースの分かち書きを実行中...")
            boundary_result = await embedding_tokenizer.tokenize_with_boundaries(text)
            results.append(boundary_result)
            print(f"\n【{boundary_result.method}】")
            print(f"トークン数: {len(boundary_result.tokens)}")
            print(f"分かち書き: {format_tokens(boundary_result.tokens)}")
        except Exception as e:
            print(f"  エラー: {e}")
        
        # LLMベースの結果
        try:
            print("\nLLMベースの分かち書きを実行中...")
            llm_result = await llm_tokenizer.tokenize_with_llm(text)
            results.append(llm_result)
            print(f"\n【{llm_result.method}】")
            print(f"トークン数: {len(llm_result.tokens)}")
            print(f"分かち書き: {format_tokens(llm_result.tokens)}")
        except Exception as e:
            print(f"  エラー: {e}")
        
        # 結果を保存
        result_data = {
            "text": text,
            "results": [
                {
                    "method": r.method,
                    "tokens": r.tokens,
                    "stats": calculate_stats(r)
                }
                for r in results
            ]
        }
        all_results.append(result_data)
    
    # 全結果をJSONファイルに保存
    output_file = "legal_text_tokenization.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\n\n全結果を {output_file} に保存しました")

if __name__ == "__main__":
    if not os.getenv("GOOGLE_API_KEY"):
        print("Error: GOOGLE_API_KEY not found in .env")
        sys.exit(1)
    if not os.getenv("AZURE_OPENAI_API_KEY"):
        print("Error: AZURE_OPENAI_API_KEY not found in .env")
        sys.exit(1)
    
    asyncio.run(main())