#!/usr/bin/env python3
"""
一般コーパスに基づくIDF値を提供するモジュール
日本語の一般的な単語の出現頻度データを保持し、
専門用語の判別に使用する
"""

import math
from typing import Dict, Optional


class GeneralCorpusIDF:
    """一般コーパスのIDF値を管理するクラス"""

    def __init__(self):
        """
        初期化
        注：実際の運用では、日本語Wikipedia等から収集した
        大規模な単語頻度データを使用すべきですが、
        ここでは代表的な一般語のみを定義しています。
        """
        # 一般的な日本語単語の文書頻度（仮想的な10万文書中の出現文書数）
        self.total_docs = 100000

        # 一般語の出現文書数（高頻度＝一般的）
        self.general_word_doc_freq = {
            # 超高頻度（一般的すぎる）
            'こと': 90000,
            'もの': 85000,
            'ため': 80000,
            'とき': 75000,
            'ところ': 70000,
            'これ': 85000,
            'それ': 80000,
            'ある': 90000,
            'する': 95000,
            'なる': 88000,
            'いる': 87000,
            'ない': 85000,
            'できる': 75000,

            # 高頻度（一般的）
            '会社': 50000,
            '時間': 55000,
            '場合': 60000,
            '方法': 45000,
            '問題': 48000,
            '必要': 52000,
            '可能': 40000,
            '情報': 45000,
            '結果': 42000,
            '関係': 38000,
            '利用': 35000,
            '対応': 33000,
            '確認': 30000,
            '実施': 28000,
            '管理': 32000,
            '開発': 30000,
            '設定': 25000,
            '作成': 27000,
            '処理': 20000,
            '機能': 22000,

            # 中頻度（やや一般的）
            'システム': 15000,
            'データ': 18000,
            'サービス': 16000,
            'エンジン': 8000,
            'プログラム': 7000,
            'ソフトウェア': 6000,
            'ネットワーク': 7500,
            'コンピュータ': 9000,
            'インターネット': 8500,
            'ファイル': 5000,
            'フォルダ': 3000,
            'メール': 12000,
            'パソコン': 10000,
            'スマートフォン': 11000,

            # 低頻度（やや専門的だが汎用）
            'アプリケーション': 4000,
            'データベース': 3500,
            'アルゴリズム': 2000,
            'インターフェース': 2500,
            'プロトコル': 1500,
            'フレームワーク': 1800,
            'ライブラリ': 1200,

            # 産業用語（中程度の専門性）
            '製造': 5000,
            '生産': 6000,
            '品質': 4500,
            '効率': 7000,
            '削減': 8000,
            '改善': 9000,
            '評価': 7500,
            '分析': 6500,
            '設計': 5500,
            '開発': 6000,
            '燃料': 3000,
            'エネルギー': 4000,
            '環境': 8000,
            '安全': 7000,

            # 数値・単位（一般的）
            '100': 30000,
            '10': 35000,
            '1': 40000,
            '2': 38000,
            '3': 36000,
            '2025': 500,
            '2024': 800,
            '80': 5000,
            '90': 4500,
            '％': 20000,
            '率': 15000,
        }

        # 専門用語の例（低頻度＝専門的）
        # これらは実際のコーパスでは稀
        self.technical_word_doc_freq = {
            # 工学系専門用語
            'クランクケース': 50,
            'ガス燃料噴射弁': 10,
            'アンモニアガス': 100,
            '混焼率': 30,
            'NOx': 200,
            'N2O': 80,
            'IMO': 150,
            'IHI': 100,
            'MAN': 120,
            'SCR': 80,
            'EGR': 70,

            # その他の専門用語の推定値
            'ターボチャージャー': 150,
            'インジェクタ': 100,
            'コモンレール': 60,
            'ピストンリング': 80,
            'カムシャフト': 90,
            'バルブタイミング': 70,
        }

        # 全単語頻度を統合
        self.doc_freq = {}
        self.doc_freq.update(self.general_word_doc_freq)
        self.doc_freq.update(self.technical_word_doc_freq)

    def get_idf(self, term: str) -> float:
        """
        指定された用語のIDF値を計算

        Args:
            term: 計算対象の用語

        Returns:
            IDF値（対数スケール）
        """
        # 文書頻度を取得（未知語はデフォルト値を使用）
        doc_freq = self.doc_freq.get(term, None)

        if doc_freq is None:
            # 未知語の処理
            term_len = len(term)
            if term_len <= 2:
                # 短い未知語は一般的と仮定
                doc_freq = 5000
            elif term_len >= 8:
                # 長い未知語は専門的と仮定
                doc_freq = 50
            else:
                # 中間的な長さは中程度の頻度
                doc_freq = 500

            # カタカナ語は専門的と仮定
            if self._is_katakana_heavy(term):
                doc_freq = min(doc_freq, 200)

            # 英数字混在は専門的
            if self._has_alphanumeric(term):
                doc_freq = min(doc_freq, 100)

        # IDF計算（スムージングあり）
        idf = math.log((self.total_docs + 1) / (doc_freq + 1))
        return idf

    def get_speciality_score(self, term: str) -> float:
        """
        用語の専門性スコアを計算（0-1の範囲に正規化）

        Args:
            term: 評価対象の用語

        Returns:
            専門性スコア（0=一般的、1=専門的）
        """
        idf = self.get_idf(term)
        # IDF値を0-1の範囲に正規化
        # IDF範囲は約0（超高頻度）から約11.5（超低頻度）
        max_idf = math.log(self.total_docs + 1)
        score = min(1.0, idf / max_idf)
        return score

    def _is_katakana_heavy(self, term: str) -> bool:
        """カタカナが多い語かチェック"""
        import re
        katakana_count = len(re.findall(r'[ァ-ヴー]', term))
        return katakana_count >= len(term) * 0.5

    def _has_alphanumeric(self, term: str) -> bool:
        """英数字を含むかチェック"""
        import re
        return bool(re.search(r'[A-Za-z0-9]', term))

    def explain_score(self, term: str) -> Dict[str, any]:
        """
        用語のスコア詳細を説明

        Args:
            term: 説明対象の用語

        Returns:
            スコアの詳細情報
        """
        doc_freq = self.doc_freq.get(term, "推定値")
        idf = self.get_idf(term)
        speciality = self.get_speciality_score(term)

        # 判定
        if speciality < 0.3:
            category = "一般語"
        elif speciality < 0.6:
            category = "準専門用語"
        else:
            category = "専門用語"

        return {
            "term": term,
            "doc_frequency": doc_freq,
            "idf_value": round(idf, 3),
            "speciality_score": round(speciality, 3),
            "category": category
        }


# シングルトン的に使用
_corpus_idf = None

def get_corpus_idf() -> GeneralCorpusIDF:
    """コーパスIDFインスタンスを取得"""
    global _corpus_idf
    if _corpus_idf is None:
        _corpus_idf = GeneralCorpusIDF()
    return _corpus_idf


if __name__ == "__main__":
    # テスト実行
    corpus = get_corpus_idf()

    test_terms = [
        "エンジン",
        "クランクケース",
        "ガス燃料噴射弁",
        "こと",
        "削減",
        "IMO",
        "アンモニアガス",
        "100",
        "混焼率"
    ]

    print("用語の専門性評価テスト:")
    print("-" * 60)
    for term in test_terms:
        info = corpus.explain_score(term)
        print(f"{term:15} | IDF: {info['idf_value']:6.3f} | "
              f"専門性: {info['speciality_score']:5.3f} | "
              f"分類: {info['category']}")