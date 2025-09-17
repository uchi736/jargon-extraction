#!/usr/bin/env python3
"""
SemRe-Rankを用いた専門用語抽出システム
統計的手法（TF-IDF、C-value）とグラフベース手法（kNN + Personalized PageRank）の統合
"""

import sys
from pathlib import Path
import asyncio
from typing import List, Dict, Optional, Tuple, Any
from collections import Counter, defaultdict
import math
import json
import pickle
import hashlib
import itertools

# プロジェクトルートをパスに追加
sys.path.insert(0, str(Path(__file__).parent.parent))

# 既存のベースクラスを継承
from statistical_extractor import StatisticalTermExtractor
from src.utils.base_extractor import Term

# 必要なライブラリ
import numpy as np
import networkx as nx
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import unicodedata

from dotenv import load_dotenv
from rich.console import Console

load_dotenv()
console = Console()


class StatisticalTermExtractorV2(StatisticalTermExtractor):
    """SemRe-Rankによる専門用語抽出器（kNN + Personalized PageRank）"""

    def __init__(
        self,
        # 既存パラメータ
        use_llm_validation: bool = True,
        min_term_length: int = 2,
        max_term_length: int = 10,
        min_frequency: int = 2,
        # SemRe-Rank用パラメータ
        k_neighbors: int = 12,
        sim_threshold: float = 0.30,
        alpha: float = 0.85,
        gamma: float = 0.7,
        beta: float = 0.3,
        w_pagerank: float = 0.6,
        embedding_model: str = "sonoisa/sentence-bert-base-ja-mean-tokens-v2",
        use_cache: bool = True,
        cache_dir: str = "cache/embeddings"
    ):
        """
        初期化

        Args:
            use_llm_validation: LLMによる検証を行うか
            min_term_length: 最小用語長
            max_term_length: 最大用語長
            min_frequency: 最小出現頻度
            k_neighbors: kNN近傍数
            sim_threshold: 類似度閾値
            alpha: PageRankダンピング係数
            gamma: Personalization指数
            beta: 共起重み係数
            w_pagerank: PageRank重み（最終スコア統合用）
            embedding_model: 使用する埋め込みモデル
            use_cache: 埋め込みキャッシュを使用するか
            cache_dir: キャッシュディレクトリ
        """
        super().__init__(
            use_llm_validation=use_llm_validation,
            min_term_length=min_term_length,
            max_term_length=max_term_length,
            min_frequency=min_frequency
        )

        # SemRe-Rankパラメータ
        self.k_neighbors = k_neighbors
        self.sim_threshold = sim_threshold
        self.alpha = alpha
        self.gamma = gamma
        self.beta = beta
        self.w_pagerank = w_pagerank
        self.use_cache = use_cache
        self.cache_dir = Path(cache_dir)

        # 埋め込みモデル
        console.print(f"[cyan]埋め込みモデルを読み込み中: {embedding_model}[/cyan]")
        self.embedder = SentenceTransformer(embedding_model)

        # キャッシュディレクトリ作成
        if use_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    async def extract_terms_from_text(
        self,
        text: str,
        metadata: Optional[Dict] = None
    ) -> List[Term]:
        """
        SemRe-Rankで専門用語を抽出

        Args:
            text: 抽出対象のテキスト
            metadata: 追加メタデータ

        Returns:
            抽出された専門用語リスト
        """
        console.print("[cyan]SemRe-Rankで専門用語を抽出中...[/cyan]")

        # 1. 統計的手法で初期候補を抽出（親クラスのメソッドを利用）
        console.print("  [yellow]1. 統計的手法で初期候補を抽出[/yellow]")
        candidates = self._extract_candidates(text)

        # 2. TF-IDFとC-valueを計算
        tfidf_scores = self._calculate_tfidf(text, list(candidates.keys()))
        cvalue_scores = self._calculate_cvalue(candidates)

        # スコアを正規化
        tfidf_normalized = self._normalize_scores(tfidf_scores)
        cvalue_normalized = self._normalize_scores(cvalue_scores)

        # 3. 初期スコアを持つTermオブジェクトを作成
        terms = []
        for term, frequency in candidates.items():
            if frequency < self.min_frequency:
                continue

            # 統計スコア（base_score）を計算
            tfidf = tfidf_normalized.get(term, 0.0)
            cvalue = cvalue_normalized.get(term, 0.0)
            base_score = (tfidf * 0.3 + cvalue * 0.7)

            terms.append(Term(
                term=term,
                definition="",
                score=base_score,
                frequency=frequency,
                contexts=self._extract_contexts(text, term),
                metadata={
                    "tfidf": tfidf,
                    "cvalue": cvalue,
                    "base_score": base_score
                }
            ))

        # スコアでソート
        terms.sort(key=lambda x: x.score, reverse=True)

        # 4. 前処理フィルタ（汎用語除去）
        console.print(f"  [yellow]2. 前処理フィルタ（候補数: {len(terms)}）[/yellow]")
        terms = self._prefilter_terms(terms)
        console.print(f"    [dim]フィルタ後: {len(terms)}件[/dim]")

        # 5. 上位N件に絞り込み（グラフ構築の効率化）
        max_graph_nodes = min(300, len(terms))
        if len(terms) > max_graph_nodes:
            terms = terms[:max_graph_nodes]
            console.print(f"    [dim]グラフ構築用に上位{max_graph_nodes}件に絞り込み[/dim]")

        if len(terms) == 0:
            return []

        # 6. 埋め込みを計算
        console.print("  [yellow]3. 埋め込みを計算[/yellow]")
        embeddings = self._compute_embeddings([t.term for t in terms])

        # 7. kNNグラフを構築
        console.print("  [yellow]4. kNNグラフを構築[/yellow]")
        graph = self._build_knn_graph(terms, embeddings)

        # 8. 軽量共起でエッジ重みを補正
        console.print("  [yellow]5. 共起情報でエッジ重みを補正[/yellow]")
        self._apply_cooccurrence_weight(graph, text)

        # 9. Personalized PageRankを実行
        console.print("  [yellow]6. Personalized PageRankを実行[/yellow]")
        pagerank_scores = self._personalized_pagerank(graph, terms)

        # 10. スコアを統合
        console.print("  [yellow]7. スコアを統合[/yellow]")
        terms = self._fuse_scores(terms, pagerank_scores)

        # 11. スコア分布から検証数を決定
        validation_count = self._determine_validation_count_by_distribution(terms)

        # 12. LLM検証（オプション）
        if self.use_llm_validation and terms:
            console.print(f"  [cyan]LLM検証対象: 上位{validation_count}件[/cyan]")
            terms = await self._validate_with_llm(terms[:validation_count], text)
            console.print(f"  [yellow]LLM検証後の候補数: {len(terms)}[/yellow]")

        # 最終結果
        final_terms = terms[:min(validation_count, 50)]
        console.print(f"  [green]最終的な専門用語数: {len(final_terms)}[/green]")
        return final_terms

    def _prefilter_terms(self, terms: List[Term]) -> List[Term]:
        """
        前処理フィルタ（汎用語・レイアウト語除去）

        Args:
            terms: フィルタ前の用語リスト

        Returns:
            フィルタ後の用語リスト
        """
        # 汎用的すぎる語のリスト（ドメインに応じて調整）
        GENERIC_TERMS = {
            'システム', 'データ', '情報', '処理', '管理', '機能', '設定',
            'ファイル', 'フォルダ', 'エリア', 'モード', 'タイプ', 'レベル',
            '状態', '結果', '対象', '内容', '項目', '要素', '部分', '全体'
        }

        # レイアウト関連語
        LAYOUT_TERMS = {
            '図', '表', 'ページ', '章', '節', '項', '参照', '以下', '上記',
            '次', '前', '後', '左', '右', '上', '下', '例', 'まとめ'
        }

        filtered = []
        for term in terms:
            # 正規化
            norm_term = unicodedata.normalize('NFKC', term.term)

            # フィルタリング
            if norm_term in GENERIC_TERMS or norm_term in LAYOUT_TERMS:
                continue

            # 1文字の語は除外
            if len(norm_term) == 1:
                continue

            filtered.append(term)

        return filtered

    def _normalize_term(self, term: str) -> str:
        """
        用語の正規化（NFKC、記号統一）

        Args:
            term: 正規化前の用語

        Returns:
            正規化後の用語
        """
        # Unicode正規化
        normalized = unicodedata.normalize('NFKC', term)

        # 記号の統一（必要に応じて追加）
        normalized = normalized.replace('−', '-')
        normalized = normalized.replace('～', '~')

        return normalized

    def _compute_embeddings(self, terms: List[str]) -> Dict[str, np.ndarray]:
        """
        用語の埋め込みを計算（キャッシュ対応）

        Args:
            terms: 用語リスト

        Returns:
            {用語: 埋め込みベクトル} の辞書
        """
        embeddings = {}
        uncached_terms = []

        # キャッシュから読み込み
        if self.use_cache:
            for term in terms:
                cache_key = self._get_cache_key(term)
                cache_path = self.cache_dir / f"{cache_key}.pkl"

                if cache_path.exists():
                    with open(cache_path, 'rb') as f:
                        embeddings[term] = pickle.load(f)
                else:
                    uncached_terms.append(term)
        else:
            uncached_terms = terms

        # 未キャッシュの用語を一括計算
        if uncached_terms:
            console.print(f"    [dim]埋め込み計算: {len(uncached_terms)}件[/dim]")

            # 正規化
            normalized_terms = [self._normalize_term(t) for t in uncached_terms]

            # バッチで埋め込み計算
            new_embeddings = self.embedder.encode(
                normalized_terms,
                convert_to_numpy=True,
                show_progress_bar=False
            )

            # L2正規化
            new_embeddings = new_embeddings / np.linalg.norm(
                new_embeddings, axis=1, keepdims=True
            )

            # 辞書に追加＆キャッシュ保存
            for term, emb in zip(uncached_terms, new_embeddings):
                embeddings[term] = emb

                if self.use_cache:
                    cache_key = self._get_cache_key(term)
                    cache_path = self.cache_dir / f"{cache_key}.pkl"
                    with open(cache_path, 'wb') as f:
                        pickle.dump(emb, f)

        return embeddings

    def _get_cache_key(self, term: str) -> str:
        """
        キャッシュキーを生成

        Args:
            term: 用語

        Returns:
            ハッシュ化されたキー
        """
        normalized = self._normalize_term(term)
        return hashlib.md5(normalized.encode()).hexdigest()

    def _build_knn_graph(
        self,
        terms: List[Term],
        embeddings: Dict[str, np.ndarray]
    ) -> nx.Graph:
        """
        kNNグラフを構築

        Args:
            terms: 用語リスト
            embeddings: 埋め込み辞書

        Returns:
            NetworkXグラフ
        """
        keys = [t.term for t in terms]
        X = np.vstack([embeddings[k] for k in keys])

        # k近傍探索
        k = min(self.k_neighbors + 1, len(keys))
        nn = NearestNeighbors(
            n_neighbors=k,
            metric='cosine'
        ).fit(X)

        distances, indices = nn.kneighbors(X)

        # グラフ作成
        G = nx.Graph()

        # ノード追加
        for i, term_obj in enumerate(terms):
            G.add_node(
                term_obj.term,
                base_score=float(term_obj.score),
                frequency=int(term_obj.frequency)
            )

        # エッジ追加（k近傍のみ）
        edge_count = 0
        for i, term_i in enumerate(terms):
            for j_pos in range(1, len(indices[i])):  # 自分自身（j_pos=0）を除く
                j = indices[i][j_pos]
                if j >= len(terms):
                    continue

                term_j = terms[j]

                # コサイン類似度（1 - コサイン距離）
                similarity = 1.0 - distances[i][j_pos]

                # 閾値チェック
                if similarity >= self.sim_threshold:
                    G.add_edge(
                        term_i.term,
                        term_j.term,
                        weight=float(similarity)
                    )
                    edge_count += 1

        console.print(
            f"    [dim]グラフ構築完了: {len(G.nodes)}ノード, "
            f"{G.number_of_edges()}エッジ[/dim]"
        )

        return G

    def _apply_cooccurrence_weight(self, graph: nx.Graph, text: str):
        """
        共起情報でエッジ重みを補正

        Args:
            graph: NetworkXグラフ
            text: 元のテキスト
        """
        # 文単位で分割
        sentences = text.replace('\n', '。').split('。')

        # 共起カウント
        cooccur_counts = defaultdict(int)
        terms = list(graph.nodes())

        for sent in sentences:
            # この文に含まれる用語を探索
            present_terms = []
            for term in terms:
                if term in sent:
                    present_terms.append(term)

            # ペアごとに共起カウント
            for t1, t2 in itertools.combinations(present_terms, 2):
                key = tuple(sorted([t1, t2]))
                cooccur_counts[key] += 1

        # エッジ重みを更新
        updated = 0
        for u, v, data in graph.edges(data=True):
            key = tuple(sorted([u, v]))
            cooccur = cooccur_counts.get(key, 0)

            if cooccur > 0:
                # log(1+共起回数)で重みを補正
                factor = 1.0 + self.beta * np.log1p(cooccur)
                data['weight'] = float(data['weight'] * factor)
                updated += 1

        console.print(f"    [dim]共起情報で{updated}エッジを補正[/dim]")

    def _personalized_pagerank(
        self,
        graph: nx.Graph,
        terms: List[Term]
    ) -> Dict[str, float]:
        """
        Personalized PageRankを実行

        Args:
            graph: NetworkXグラフ
            terms: 用語リスト

        Returns:
            {用語: PageRankスコア} の辞書
        """
        # Personalizationベクトルを作成（base_score^gamma）
        personalization = {}
        for term in terms:
            if term.term in graph.nodes:
                base_score = graph.nodes[term.term]['base_score']
                personalization[term.term] = base_score ** self.gamma

        # 正規化
        total = sum(personalization.values()) or 1.0
        personalization = {k: v/total for k, v in personalization.items()}

        # PageRank実行
        try:
            pagerank = nx.pagerank(
                graph,
                alpha=self.alpha,
                personalization=personalization,
                weight='weight',
                max_iter=100,
                tol=1e-4
            )
            console.print(f"    [dim]PageRank収束[/dim]")
        except nx.PowerIterationFailedConvergence:
            console.print("[yellow]    PageRank未収束、結果を使用[/yellow]")
            pagerank = nx.pagerank(
                graph,
                alpha=self.alpha,
                personalization=personalization,
                weight='weight',
                max_iter=100,
                tol=1e-3
            )

        return pagerank

    def _fuse_scores(
        self,
        terms: List[Term],
        pagerank_scores: Dict[str, float]
    ) -> List[Term]:
        """
        統計スコアとPageRankスコアを統合

        Args:
            terms: 用語リスト
            pagerank_scores: PageRankスコア辞書

        Returns:
            スコア更新済みの用語リスト
        """
        # スコアを配列化
        base_scores = np.array([t.score for t in terms])
        pr_scores = np.array([pagerank_scores.get(t.term, 0.0) for t in terms])

        # Min-Max正規化
        def normalize_array(arr):
            if len(arr) == 0:
                return arr
            min_val = arr.min()
            max_val = arr.max()
            if max_val == min_val:
                return np.full_like(arr, 0.5)
            return (arr - min_val) / (max_val - min_val)

        base_normalized = normalize_array(base_scores)
        pr_normalized = normalize_array(pr_scores)

        # 重み付き統合
        fused = self.w_pagerank * pr_normalized + (1 - self.w_pagerank) * base_normalized

        # スコア更新
        for term, score in zip(terms, fused):
            term.score = float(score)
            term.metadata['pagerank'] = pagerank_scores.get(term.term, 0.0)

        # 再ソート
        terms.sort(key=lambda x: x.score, reverse=True)

        return terms


async def main():
    """メイン処理"""
    if len(sys.argv) < 2:
        console.print("[red]使用法: python statistical_extractor_V2.py <input_path> [output.json][/red]")
        sys.exit(1)

    input_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2] if len(sys.argv) > 2 else "output/semre_rank_terms.json")

    # 抽出器を初期化
    extractor = StatisticalTermExtractorV2(
        use_llm_validation=True,
        min_frequency=2,
        k_neighbors=12,
        sim_threshold=0.30,
        w_pagerank=0.6
    )

    # 処理実行
    if input_path.is_file():
        result = await extractor.extract_from_file(input_path)
        results = [result]
    else:
        results = await extractor.extract_from_directory(input_path)

    if results:
        # 結果を表示
        all_terms = []
        for result in results:
            all_terms.extend(result.terms)

        merged_terms = extractor.merge_terms([all_terms])
        extractor.display_terms(merged_terms)

        # 保存
        extractor.save_results(results, output_path)

        # 統計を表示
        stats = extractor.get_statistics(results)
        console.print(f"\n[green]処理統計:[/green]")
        for key, value in stats.items():
            console.print(f"  {key}: {value}")


if __name__ == "__main__":
    asyncio.run(main())