#!/usr/bin/env python3
"""
Azure OpenAI Embedding テストスクリプト
"""

import asyncio
import os
import time
from dotenv import load_dotenv
from langchain_openai import AzureOpenAIEmbeddings

load_dotenv()

async def test_azure_embedding():
    """Azure OpenAI埋め込みのテスト"""
    
    # 環境変数を確認
    print("環境変数の確認:")
    print(f"AZURE_OPENAI_ENDPOINT: {os.getenv('AZURE_OPENAI_ENDPOINT')}")
    print(f"AZURE_OPENAI_API_VERSION: {os.getenv('AZURE_OPENAI_API_VERSION')}")
    print(f"AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME: {os.getenv('AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME')}")
    print(f"AZURE_OPENAI_API_KEY: {'設定済み' if os.getenv('AZURE_OPENAI_API_KEY') else '未設定'}")
    print()
    
    # Embeddingを初期化
    embeddings = AzureOpenAIEmbeddings(
        azure_deployment=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME"),
        openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY")
    )
    
    # テストテキスト
    test_texts = [
        "自然言語処理",
        "機械学習",
        "深層学習"
    ]
    
    print("埋め込みベクトル生成テスト:")
    print("-" * 50)
    
    for text in test_texts:
        print(f"\nテキスト: {text}")
        start_time = time.time()
        
        try:
            # 埋め込みベクトルを生成
            vector = await embeddings.aembed_query(text)
            elapsed = time.time() - start_time
            
            print(f"  ベクトル次元数: {len(vector)}")
            print(f"  処理時間: {elapsed:.3f}秒")
            print(f"  ベクトルの最初の5要素: {vector[:5]}")
            
        except Exception as e:
            print(f"  エラー: {e}")
    
    print("\n" + "=" * 50)
    print("類似度計算テスト:")
    
    try:
        # 複数テキストの埋め込み
        texts = ["機械学習", "深層学習", "統計学習"]
        print(f"テキスト: {texts}")
        
        start_time = time.time()
        vectors = await embeddings.aembed_documents(texts)
        elapsed = time.time() - start_time
        
        print(f"処理時間: {elapsed:.3f}秒")
        print(f"生成されたベクトル数: {len(vectors)}")
        
        # 簡単な類似度計算（コサイン類似度）
        import numpy as np
        from sklearn.metrics.pairwise import cosine_similarity
        
        similarity_matrix = cosine_similarity(vectors)
        print("\n類似度行列:")
        for i, text1 in enumerate(texts):
            for j, text2 in enumerate(texts):
                if i < j:
                    print(f"  {text1} - {text2}: {similarity_matrix[i][j]:.3f}")
                    
    except Exception as e:
        print(f"エラー: {e}")

if __name__ == "__main__":
    asyncio.run(test_azure_embedding())