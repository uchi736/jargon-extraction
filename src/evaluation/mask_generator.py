#!/usr/bin/env python3
"""
MASK文章生成戦略：専門用語評価のための文脈作成
"""
import random
import re
from typing import List, Dict, Tuple

class MaskTextGenerator:
    """MASK文章を生成する戦略クラス"""
    
    def __init__(self):
        # 文脈テンプレート（前・後）
        self.context_templates = {
            'definition': {
                'prefix': [
                    '[MASK]とは',
                    '[MASK]は',
                    '[MASK]という用語は',
                    'ここでいう[MASK]は',
                    '専門用語の[MASK]は',
                ],
                'suffix': [
                    'を意味する。',
                    'である。',
                    'を指す。',
                    'と定義される。',
                    'のことである。',
                ]
            },
            'explanation': {
                'prefix': [
                    'この[MASK]について',
                    '[MASK]を',
                    '重要な[MASK]を',
                    '[MASK]の概念を',
                    '技術的な[MASK]を',
                ],
                'suffix': [
                    '説明します。',
                    '解説する。',
                    '理解することが重要だ。',
                    '検討する必要がある。',
                    '分析してみよう。',
                ]
            },
            'calculation': {
                'prefix': [
                    '[MASK]を',
                    '[MASK]の値を',
                    '正確な[MASK]を',
                    '[MASK]について',
                    'システムの[MASK]を',
                ],
                'suffix': [
                    '計算する。',
                    '算出する。',
                    '求める。',
                    '評価する。',
                    '測定する。',
                ]
            },
            'application': {
                'prefix': [
                    '[MASK]を使用して',
                    '[MASK]による',
                    '[MASK]を適用し',
                    '[MASK]の技術で',
                    '[MASK]方式により',
                ],
                'suffix': [
                    '実装する。',
                    '処理を行う。',
                    '最適化する。',
                    '改善する。',
                    '実現する。',
                ]
            }
        }
        
        # ドメイン別の文脈パターン
        self.domain_patterns = {
            'engineering': {
                'keywords': ['設計', '構造', '性能', '効率', '機構'],
                'templates': [
                    '{keyword}における[MASK]の重要性',
                    '[MASK]による{keyword}の最適化',
                    '{keyword}と[MASK]の関係',
                ]
            },
            'chemistry': {
                'keywords': ['反応', '濃度', '触媒', '化合物', '分子'],
                'templates': [
                    '{keyword}中の[MASK]',
                    '[MASK]を用いた{keyword}',
                    '{keyword}における[MASK]の役割',
                ]
            },
            'ai_ml': {
                'keywords': ['学習', 'モデル', 'データ', 'アルゴリズム', '精度'],
                'templates': [
                    '{keyword}に基づく[MASK]',
                    '[MASK]を使った{keyword}',
                    '{keyword}における[MASK]の適用',
                ]
            }
        }
    
    def generate_mask_contexts(self, term: str, text: str = None, 
                              num_contexts: int = 5) -> List[Dict]:
        """
        複数のMASK文脈を生成
        
        Args:
            term: マスクする専門用語
            text: 元のテキスト（あれば実際の文脈を使用）
            num_contexts: 生成する文脈数
        
        Returns:
            List[Dict]: MASK文脈のリスト
        """
        contexts = []
        
        # 1. 実際のテキストからの文脈抽出（最優先）
        if text:
            real_contexts = self._extract_real_contexts(term, text)
            contexts.extend(real_contexts[:num_contexts//2])
        
        # 2. テンプレートベースの文脈生成
        template_contexts = self._generate_template_contexts(term)
        contexts.extend(template_contexts[:num_contexts//2])
        
        # 3. ドメイン特化の文脈生成
        domain_contexts = self._generate_domain_contexts(term)
        contexts.extend(domain_contexts[:num_contexts//4])
        
        # 4. 統計的文脈生成（n-gram的アプローチ）
        if text:
            statistical_contexts = self._generate_statistical_contexts(term, text)
            contexts.extend(statistical_contexts[:num_contexts//4])
        
        # 重複除去と制限
        seen = set()
        unique_contexts = []
        for ctx in contexts:
            if ctx['masked_text'] not in seen:
                seen.add(ctx['masked_text'])
                unique_contexts.append(ctx)
                if len(unique_contexts) >= num_contexts:
                    break
        
        return unique_contexts
    
    def _extract_real_contexts(self, term: str, text: str, 
                               window_size: int = 50) -> List[Dict]:
        """実際のテキストから文脈を抽出"""
        contexts = []
        
        # 用語の出現箇所を検索
        for match in re.finditer(re.escape(term), text):
            start = match.start()
            end = match.end()
            
            # 前後の文脈を取得
            prefix_start = max(0, start - window_size)
            suffix_end = min(len(text), end + window_size)
            
            prefix = text[prefix_start:start]
            suffix = text[end:suffix_end]
            
            # 文の境界で切る
            if '。' in prefix:
                prefix = prefix[prefix.rfind('。')+1:]
            if '。' in suffix:
                suffix = suffix[:suffix.find('。')+1]
            
            masked_text = prefix + "[MASK]" + suffix
            
            contexts.append({
                'type': 'real',
                'masked_text': masked_text.strip(),
                'original_text': prefix + term + suffix,
                'prefix': prefix.strip(),
                'suffix': suffix.strip(),
                'term': term
            })
        
        return contexts
    
    def _generate_template_contexts(self, term: str) -> List[Dict]:
        """テンプレートベースの文脈生成"""
        contexts = []
        
        for context_type, templates in self.context_templates.items():
            # 各タイプから1つずつ生成
            prefix = random.choice(templates['prefix'])
            suffix = random.choice(templates['suffix'])
            
            masked_text = prefix + suffix
            original_text = masked_text.replace('[MASK]', term)
            
            contexts.append({
                'type': f'template_{context_type}',
                'masked_text': masked_text,
                'original_text': original_text,
                'prefix': prefix.replace('[MASK]', ''),
                'suffix': suffix,
                'term': term
            })
        
        return contexts
    
    def _generate_domain_contexts(self, term: str) -> List[Dict]:
        """ドメイン特化の文脈生成"""
        contexts = []
        
        # 用語から推定されるドメインを判定
        domain = self._detect_domain(term)
        
        if domain in self.domain_patterns:
            pattern_info = self.domain_patterns[domain]
            
            for template in pattern_info['templates'][:2]:
                keyword = random.choice(pattern_info['keywords'])
                masked_text = template.format(keyword=keyword)
                original_text = masked_text.replace('[MASK]', term)
                
                contexts.append({
                    'type': f'domain_{domain}',
                    'masked_text': masked_text,
                    'original_text': original_text,
                    'domain': domain,
                    'keyword': keyword,
                    'term': term
                })
        
        return contexts
    
    def _generate_statistical_contexts(self, term: str, text: str) -> List[Dict]:
        """統計的な共起パターンから文脈生成"""
        contexts = []
        
        # 用語の前後によく現れる語を収集
        cooccurrences = self._find_cooccurrences(term, text)
        
        for direction, words in cooccurrences.items():
            if words:
                common_word = words[0][0]  # 最頻出語
                
                if direction == 'before':
                    masked_text = f"{common_word}[MASK]"
                else:
                    masked_text = f"[MASK]{common_word}"
                
                original_text = masked_text.replace('[MASK]', term)
                
                contexts.append({
                    'type': 'statistical',
                    'masked_text': masked_text,
                    'original_text': original_text,
                    'direction': direction,
                    'cooccurrence': common_word,
                    'term': term
                })
        
        return contexts
    
    def _detect_domain(self, term: str) -> str:
        """用語からドメインを推定"""
        # 簡易的なドメイン判定
        engineering_keywords = ['エンジン', '機構', '設計', 'システム', '制御']
        chemistry_keywords = ['化合物', '反応', '触媒', '分子', '濃度', '燃料', '燃焼']
        ai_ml_keywords = ['学習', 'モデル', 'データ', 'アルゴリズム', 'ネットワーク']
        
        for keyword in engineering_keywords:
            if keyword in term:
                return 'engineering'
        
        for keyword in chemistry_keywords:
            if keyword in term:
                return 'chemistry'
        
        for keyword in ai_ml_keywords:
            if keyword in term:
                return 'ai_ml'
        
        return 'general'
    
    def _find_cooccurrences(self, term: str, text: str, 
                           window: int = 5) -> Dict[str, List[Tuple[str, int]]]:
        """用語の前後に現れる語を統計的に収集"""
        from collections import Counter
        
        words = text.split()
        before_words = Counter()
        after_words = Counter()
        
        for i, word in enumerate(words):
            if term in word:
                # 前の語
                if i > 0:
                    for j in range(max(0, i-window), i):
                        before_words[words[j]] += 1
                
                # 後の語
                if i < len(words) - 1:
                    for j in range(i+1, min(len(words), i+window+1)):
                        after_words[words[j]] += 1
        
        return {
            'before': before_words.most_common(3),
            'after': after_words.most_common(3)
        }
    
    def evaluate_mask_quality(self, masked_text: str, term: str) -> float:
        """MASK文脈の品質を評価（0-1）"""
        score = 1.0
        
        # 長さチェック
        if len(masked_text) < 10:
            score *= 0.5
        elif len(masked_text) > 200:
            score *= 0.8
        
        # [MASK]の位置チェック（文頭・文末は避ける）
        mask_pos = masked_text.find('[MASK]')
        relative_pos = mask_pos / len(masked_text)
        if relative_pos < 0.1 or relative_pos > 0.9:
            score *= 0.7
        
        # 文法的完全性（句読点の有無）
        if '。' in masked_text or '、' in masked_text:
            score *= 1.1
        
        # コンテキストの豊富さ（単語数）
        word_count = len(masked_text.split())
        if word_count < 5:
            score *= 0.6
        elif word_count > 15:
            score *= 1.2
        
        return min(1.0, score)


# デモ用関数
def demo_mask_generation():
    """MASK生成のデモ"""
    generator = MaskTextGenerator()
    
    # テスト用テキスト
    text = """
    理論空燃比は内燃機関において重要な指標である。
    アンモニア燃料の理論空燃比を正確に計算することで、
    エンジンの効率を最適化できる。
    理論空燃比が適切でない場合、不完全燃焼が発生する。
    """
    
    term = "理論空燃比"
    
    print(f"=== 「{term}」のMASK文脈生成 ===\n")
    
    contexts = generator.generate_mask_contexts(term, text, num_contexts=8)
    
    for i, ctx in enumerate(contexts, 1):
        quality = generator.evaluate_mask_quality(ctx['masked_text'], term)
        print(f"{i}. タイプ: {ctx['type']}")
        print(f"   MASK文: {ctx['masked_text'][:100]}")
        print(f"   元文: {ctx['original_text'][:100]}")
        print(f"   品質スコア: {quality:.2f}")
        print()
    
    # ドメイン別の例
    print("\n=== ドメイン特化の例 ===")
    domain_terms = {
        'engineering': 'ディーゼルエンジン',
        'chemistry': 'アンモニア燃料',
        'ai_ml': 'ニューラルネットワーク'
    }
    
    for domain, term in domain_terms.items():
        contexts = generator._generate_domain_contexts(term)
        if contexts:
            print(f"\n{domain}: {term}")
            print(f"  → {contexts[0]['masked_text']}")

if __name__ == "__main__":
    demo_mask_generation()