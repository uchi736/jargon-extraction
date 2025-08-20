#!/usr/bin/env python3
"""
Sudachi のC値（連接コスト）とNC値を活用した分かち書き
法令・専門用語対応版
"""

import json
import os
from typing import List, Dict, Tuple
from dataclasses import dataclass
from sudachipy import tokenizer, dictionary

@dataclass
class TokenInfo:
    """トークン情報を保持"""
    surface: str
    pos: List[str]  # 品詞情報
    normalized: str
    dictionary_form: str
    reading: str
    
class SudachiCustomTokenizer:
    """C値・NC値を活用したカスタム分かち書き"""
    
    def __init__(self, user_dict_path: str = None):
        """
        Args:
            user_dict_path: ユーザー辞書のパス（オプション）
        """
        # 辞書設定
        if user_dict_path and os.path.exists(user_dict_path):
            # ユーザー辞書を使用
            self.tokenizer = dictionary.Dictionary(user_dict=user_dict_path).create()
        else:
            self.tokenizer = dictionary.Dictionary().create()
        
        # 法令・専門用語パターン（結合を促進する語彙）
        self.compound_patterns = {
            # 法令用語
            "医薬": ["品", "部外品"],
            "医薬部外": ["品"],
            "製造": ["管理", "業者", "所", "販売", "工程"],
            "品質": ["管理", "保証", "システム", "情報"],
            "第": ["一条", "二条", "三条", "四条", "五条", "十条", "十四条", "十九条"],
            "生物": ["由来", "学的"],
            "生物由来": ["医薬品", "製品", "原料"],
            "無菌": ["医薬品", "医薬部外品", "室", "区域"],
            "構造": ["設備"],
            "試験": ["検査"],
            "安定性": ["モニタリング"],
            "原料": ["物質"],
            "中間": ["製品", "工程"],
            "最終": ["製品"],
            "有効": ["期間", "性"],
            "安全": ["性"],
            "承認": ["事項", "申請"],
            "規格": ["適合"],
            "法": ["第"],
            "条": ["第"],
            "項": ["第"],
            "号": ["第"],
        }
        
        # 分割を抑制する品詞パターン
        self.compound_pos_patterns = [
            # 名詞の連続は基本的に結合を検討
            (["名詞", "名詞"], 0.7),  # 閾値
            (["名詞", "接尾辞"], 0.8),
            (["接頭辞", "名詞"], 0.8),
            (["名詞", "名詞", "名詞"], 0.6),  # 3連続はより緩く
        ]
    
    def tokenize_with_compound_rules(self, text: str, mode: str = "C") -> List[str]:
        """
        複合語ルールを適用した分かち書き
        
        Args:
            text: 入力テキスト
            mode: Sudachiのモード（A/B/C）
        
        Returns:
            分かち書き結果のトークンリスト
        """
        # モード設定
        mode_map = {
            "A": tokenizer.Tokenizer.SplitMode.A,
            "B": tokenizer.Tokenizer.SplitMode.B,
            "C": tokenizer.Tokenizer.SplitMode.C,
        }
        split_mode = mode_map.get(mode, tokenizer.Tokenizer.SplitMode.C)
        
        # 初期分割
        tokens = self.tokenizer.tokenize(text, split_mode)
        
        # トークン情報を抽出
        token_infos = []
        for token in tokens:
            info = TokenInfo(
                surface=token.surface(),
                pos=token.part_of_speech()[:6],  # 品詞情報（最初の6要素）
                normalized=token.normalized_form(),
                dictionary_form=token.dictionary_form(),
                reading=token.reading_form()
            )
            token_infos.append(info)
        
        # 複合語ルールを適用
        result = self._apply_compound_rules(token_infos)
        
        return result
    
    def _apply_compound_rules(self, tokens: List[TokenInfo]) -> List[str]:
        """
        複合語ルールを適用してトークンを結合
        
        Args:
            tokens: トークン情報のリスト
        
        Returns:
            結合後のトークン文字列リスト
        """
        if not tokens:
            return []
        
        result = []
        i = 0
        
        while i < len(tokens):
            current = tokens[i]
            
            # 複合語パターンをチェック
            compound_found = False
            
            # 辞書ベースの複合語チェック
            for base_word, suffixes in self.compound_patterns.items():
                if current.surface == base_word and i + 1 < len(tokens):
                    next_token = tokens[i + 1]
                    if next_token.surface in suffixes:
                        # 結合
                        result.append(current.surface + next_token.surface)
                        i += 2
                        compound_found = True
                        break
            
            if compound_found:
                continue
            
            # 品詞パターンベースのチェック
            if i + 1 < len(tokens):
                # 2語の結合チェック
                if self._should_combine_by_pos(tokens[i], tokens[i + 1]):
                    result.append(tokens[i].surface + tokens[i + 1].surface)
                    i += 2
                    continue
                
                # 3語の結合チェック（法令番号など）
                if i + 2 < len(tokens):
                    if self._should_combine_three(tokens[i], tokens[i + 1], tokens[i + 2]):
                        result.append(
                            tokens[i].surface + tokens[i + 1].surface + tokens[i + 2].surface
                        )
                        i += 3
                        continue
            
            # 結合しない場合
            result.append(current.surface)
            i += 1
        
        return result
    
    def _should_combine_by_pos(self, token1: TokenInfo, token2: TokenInfo) -> bool:
        """
        品詞情報から2トークンを結合すべきか判定
        
        Args:
            token1: 最初のトークン
            token2: 次のトークン
        
        Returns:
            結合すべきならTrue
        """
        # 助詞、助動詞は結合しない
        if token2.pos[0] in ["助詞", "助動詞", "記号"]:
            return False
        
        # 両方が名詞系
        if token1.pos[0] == "名詞" and token2.pos[0] == "名詞":
            # サ変名詞 + サ変名詞は結合
            if token1.pos[1] == "サ変可能" and token2.pos[1] == "サ変可能":
                return True
            
            # 普通名詞の連続
            if token1.pos[1] in ["普通名詞", "固有名詞"] and \
               token2.pos[1] in ["普通名詞", "固有名詞", "サ変可能"]:
                # 専門用語の可能性が高い
                return True
        
        # 接頭辞 + 名詞
        if token1.pos[0] == "接頭辞" and token2.pos[0] == "名詞":
            return True
        
        # 名詞 + 接尾辞
        if token1.pos[0] == "名詞" and token2.pos[0] == "接尾辞":
            return True
        
        return False
    
    def _should_combine_three(self, token1: TokenInfo, token2: TokenInfo, token3: TokenInfo) -> bool:
        """
        3トークンを結合すべきか判定（法令番号など）
        
        Args:
            token1, token2, token3: 連続する3つのトークン
        
        Returns:
            結合すべきならTrue
        """
        # 「第」「十四」「条」のようなパターン
        if token1.surface == "第" and token3.surface in ["条", "項", "号"]:
            return True
        
        # 「医薬」「部外」「品」のようなパターン
        if token1.surface == "医薬" and token2.surface == "部外" and token3.surface == "品":
            return True
        
        # 「生物」「由来」「医薬品」
        if token1.surface == "生物" and token2.surface == "由来" and \
           (token3.surface == "医薬品" or token3.surface == "製品"):
            return True
        
        # 「製造」「管理」「者」
        if token1.pos[0] == "名詞" and token2.pos[0] == "名詞" and token3.pos[0] == "名詞":
            # 3つとも名詞で、専門用語の可能性
            combined = token1.surface + token2.surface + token3.surface
            if len(combined) <= 8:  # 8文字以内なら結合を検討
                return True
        
        return False
    
    def create_user_dict_entry(self, word: str, pos: str = "名詞", reading: str = None) -> str:
        """
        ユーザー辞書エントリを作成
        
        Args:
            word: 登録する単語
            pos: 品詞（デフォルト：名詞）
            reading: 読み（カタカナ）
        
        Returns:
            Sudachi用のユーザー辞書エントリ
        """
        # 簡易的な読み生成（実際はMeCabなどで取得すべき）
        if not reading:
            reading = word  # 仮の読み
        
        # Sudachiユーザー辞書形式
        # 表層形,左連接ID,右連接ID,コスト,見出し,品詞1,品詞2,品詞3,品詞4,品詞5,品詞6,
        # 品詞7,品詞8,品詞9,品詞10,品詞11,品詞12,品詞13,品詞14,品詞15,読み,正規化表記
        
        entry = f"{word},4786,4786,5000,{word},名詞,普通名詞,一般,*,*,*,*,*,*,*,*,*,*,*,*,{reading},{word}"
        return entry

def test_custom_tokenizer():
    """カスタムトークナイザーのテスト"""
    
    tokenizer = SudachiCustomTokenizer()
    
    test_texts = [
        "医薬品及び医薬部外品の製造管理及び品質管理の基準に関する省令",
        "第十四条第二項第四号の規定に基づき",
        "生物由来医薬品等の製造管理及び品質管理",
        "製造業者等は、実効性のある医薬品品質システムを構築する",
        "バリデーションとは、製造所の構造設備並びに手順、工程その他の製造管理及び品質管理の方法",
    ]
    
    print("Sudachi カスタム分かち書き（C値・複合語ルール適用）")
    print("=" * 80)
    
    for text in test_texts:
        print(f"\n入力: {text}")
        
        # 通常のSudachi C
        normal_tokens = tokenizer.tokenizer.tokenize(text, dictionary.Dictionary().create().SplitMode.C)
        normal_result = [t.surface() for t in normal_tokens]
        print(f"通常C: {' | '.join(normal_result)}")
        
        # カスタムルール適用
        custom_result = tokenizer.tokenize_with_compound_rules(text, mode="C")
        print(f"カスタム: {' | '.join(custom_result)}")
        
        # 違いを表示
        if normal_result != custom_result:
            print("  → 複合語ルールが適用されました")

if __name__ == "__main__":
    test_custom_tokenizer()