#!/usr/bin/env python3
"""
分かち書きの簡易テストスクリプト
"""

import os
import sys
from dotenv import load_dotenv
from sudachipy import tokenizer, dictionary

load_dotenv()

def test_sudachi():
    """Sudachiの基本テスト"""
    print("Sudachi分かち書きテスト")
    print("=" * 50)
    
    # テストテキスト
    test_text = "自然言語処理技術の最新動向について"
    
    # tokenizerを作成
    tok = dictionary.Dictionary().create()
    
    # 各モードで分かち書き
    for mode_name, mode in [
        ("A", tokenizer.Tokenizer.SplitMode.A),
        ("B", tokenizer.Tokenizer.SplitMode.B),
        ("C", tokenizer.Tokenizer.SplitMode.C)
    ]:
        tokens = tok.tokenize(test_text, mode)
        token_list = [token.surface() for token in tokens]
        print(f"\nMode {mode_name}: {' | '.join(token_list)}")
        print(f"トークン数: {len(token_list)}")

if __name__ == "__main__":
    test_sudachi()