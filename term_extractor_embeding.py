#!/usr/bin/env python3
"""term_extractor_lcel.py
LCEL記法版 専門用語・類義語辞書生成（SudachiPy + RAG統合版）
------------------------------------------------
* LangChain Expression Language (LCEL) でチェイン構築
* SudachiPyとN-gramによる候補語生成の前処理を追加
* Google Embedding APIとChromaDBによるRAG実装
* LangSmithによる処理トレース対応
* 構造化出力 (Pydantic) で型安全性確保
* `.env` から `GOOGLE_API_KEY` 読み込み

Quick Start::
    python -m venv .venv && .venv/Scripts/activate   # ← Windows
    pip install -U langchain langchain-google-genai \
                 langchain-core pydantic python-dotenv \
                 unstructured[all] pypdf python-docx \
                 sudachipy sudachidict_core chromadb

    echo GOOGLE_API_KEY=YOUR_KEY > .env
    echo LANGCHAIN_TRACING_V2=true >> .env
    echo LANGCHAIN_API_KEY=YOUR_LANGSMITH_KEY >> .env
    echo LANGCHAIN_PROJECT=term-extraction >> .env
    
    python term_extractor_lcel.py ./input ./output/dictionary.json
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from sudachipy import tokenizer, dictionary
import chromadb
from chromadb.utils import embedding_functions

# ── ENV ───────────────────────────────────────────
load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    sys.exit("[ERROR] .env に GOOGLE_API_KEY を設定してください")

# ── LangChain imports ─────────────────────────────
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import (
    PyPDFLoader,
    UnstructuredFileLoader,
    Docx2txtLoader,
    TextLoader,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import (
    RunnablePassthrough,
    RunnableLambda,
    RunnableParallel,
)
from langchain_core.documents import Document

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# LangSmith設定の確認（ログ出力の後に移動）
if os.getenv("LANGCHAIN_TRACING_V2", "").lower() == "true":
    if not os.getenv("LANGCHAIN_API_KEY"):
        logger.warning("LANGCHAIN_API_KEY not set. LangSmith tracing disabled.")
        os.environ["LANGCHAIN_TRACING_V2"] = "false"
    else:
        logger.info(f"LangSmith tracing enabled - Project: {os.getenv('LANGCHAIN_PROJECT', 'default')}")

# ── SudachiPy Setup ───────────────────────────────
# グローバルなtokenizerは作成せず、必要時に作成する
sudachi_mode = tokenizer.Tokenizer.SplitMode.A

# ── Vector Store Components ──────────────────────
class VectorStore:
    """ChromaDBを使用したベクトルストア"""
    
    def __init__(self):
        self.client = chromadb.Client()
        self.collection = None
        self.chunk_metadata = {}
        
    async def initialize(self, chunks: List[str], chunk_ids: List[str]):
        """チャンクをエンベディング化してベクトルストアに保存"""
        try:
            # 既存のコレクションがあれば削除
            try:
                self.client.delete_collection("term_extraction_chunks")
            except:
                pass
            
            # 新しいコレクションを作成
            self.collection = self.client.create_collection(
                name="term_extraction_chunks",
                embedding_function=None  # 手動でエンベディングを管理
            )
            
            # バッチサイズを設定（API制限対策）
            batch_size = 10
            all_embeddings = []
            
            logger.info(f"Creating embeddings for {len(chunks)} chunks...")
            
            # エンベディングをバッチで生成
            for i in range(0, len(chunks), batch_size):
                batch_chunks = chunks[i:i+batch_size]
                
                # Google Embedding APIを使用
                batch_embeddings = await embeddings.aembed_documents(batch_chunks)
                all_embeddings.extend(batch_embeddings)
                
                # レート制限対策
                if i + batch_size < len(chunks):
                    await asyncio.sleep(2)
                    
                logger.info(f"Embedded {min(i + batch_size, len(chunks))}/{len(chunks)} chunks")
            
            # ベクトルストアに追加
            self.collection.add(
                embeddings=all_embeddings,
                documents=chunks,
                ids=chunk_ids,
                metadatas=[{"chunk_id": cid, "text": chunk[:100]} for cid, chunk in zip(chunk_ids, chunks)]
            )
            
            # メタデータを保存
            for idx, (chunk_id, chunk) in enumerate(zip(chunk_ids, chunks)):
                self.chunk_metadata[chunk_id] = {
                    "text": chunk,
                    "index": idx
                }
                
            logger.info("Vector store initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize vector store: {e}")
            raise
    
    async def search_similar_chunks(self, query_text: str, current_chunk_id: str, n_results: int = 3) -> str:
        """類似チャンクを検索して関連文脈として返す"""
        try:
            if not self.collection:
                return "関連情報なし"
            
            # クエリテキストのエンベディングを生成
            query_embedding = await embeddings.aembed_query(query_text)
            
            # 類似チャンクを検索（現在のチャンクは除外）
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results + 1  # 自分自身が含まれる可能性があるため+1
            )
            
            # 結果を整形
            related_contexts = []
            for i, (doc, chunk_id, distance) in enumerate(zip(
                results['documents'][0], 
                results['ids'][0],
                results['distances'][0]
            )):
                # 自分自身のチャンクはスキップ
                if chunk_id == current_chunk_id:
                    continue
                    
                # 類似度が閾値以上のもののみ使用（距離が小さいほど類似）
                if distance < 0.5:  # 閾値は調整可能
                    related_contexts.append(f"[関連文脈 {len(related_contexts)+1}]\n{doc}")
                    
                if len(related_contexts) >= n_results:
                    break
            
            if not related_contexts:
                return "関連情報なし"
                
            return "\n\n".join(related_contexts)
            
        except Exception as e:
            logger.error(f"Error searching similar chunks: {e}")
            return "関連情報なし"

# グローバルなベクトルストアインスタンス
vector_store = VectorStore()

# ── Pydantic Models for Structured Output ────────
class Term(BaseModel):
    """専門用語の構造"""
    headword: str = Field(description="専門用語の見出し語")
    synonyms: List[str] = Field(default_factory=list, description="類義語・別名のリスト")
    definition: str = Field(description="30-50字程度の簡潔な定義")
    category: Optional[str] = Field(default=None, description="カテゴリ名")

class TermList(BaseModel):
    """用語リストの構造"""
    terms: List[Term] = Field(default_factory=list, description="専門用語のリスト")

# ── LLM Setup ─────────────────────────────────────
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0.1,
    google_api_key=API_KEY,
)

# ── Embeddings Setup ──────────────────────────────
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004",
    google_api_key=API_KEY,
    task_type="RETRIEVAL_DOCUMENT"
)

# ── Output Parser ─────────────────────────────────
json_parser = JsonOutputParser(pydantic_object=TermList)

# ── Prompts ───────────────────────────────────────
validation_prompt = ChatPromptTemplate.from_messages([
    ("system", """あなたは、提供されたテキストの文脈と候補リストを基に、専門用語を厳密に検証する専門家です。
候補リストの中から、与えられたテキストの文脈において専門用語として成立するものだけを選び出してください。
選んだ用語について、定義、類義語、カテゴリをJSON形式で返してください。
**重要：候補リストに存在しない用語を新たに追加してはいけません。**

関連する文脈情報も提供される場合は、それを参考にして：
- より正確な定義を作成
- 類義語を発見
- 適切なカテゴリ分類
を行ってください。

一般的すぎる単語（例：システム、データ、情報、処理、管理）は除外し、その分野特有の専門用語のみを選択してください。

{format_instructions}"""),
    ("user", """以下のテキストと候補リストを参考に、専門用語をJSON形式で返してください。

## テキスト本文:
{text}

## 関連する文脈情報:
{related_contexts}

## 候補リスト:
{candidates}
"""),
]).partial(format_instructions=json_parser.get_format_instructions())

consolidate_prompt = ChatPromptTemplate.from_messages([
    ("system", """用語一覧の重複を統合してください。
同じ意味の用語は1つにまとめ、類義語はsynonymsに含めてください。
必ず以下の形式の有効なJSONのみを返してください：

{format_instructions}"""),
    ("user", "{terms_json}"),
]).partial(format_instructions=json_parser.get_format_instructions())

# ── Document Processing Components ────────────────
SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size=2000,
    chunk_overlap=200,
    keep_separator=True,
    separators=["\n\n", "。", "\n", " "],
)

LOADER_MAP = {
    ".pdf": PyPDFLoader,
    ".docx": Docx2txtLoader,
    ".doc": Docx2txtLoader,
    ".txt": TextLoader,
    ".md": UnstructuredFileLoader,
    ".html": UnstructuredFileLoader,
    ".htm": UnstructuredFileLoader,
}

# ── Helper Functions ──────────────────────────────

def load_document(file_path: Path) -> List[Document]:
    """ファイルパスからドキュメントをロード"""
    try:
        loader_cls = LOADER_MAP.get(file_path.suffix.lower(), TextLoader)
        logger.info(f"Loading {file_path.name} with {loader_cls.__name__}")
        return loader_cls(str(file_path)).load()
    except Exception as e:
        logger.error(f"Failed to load {file_path}: {e}")
        return []

def split_documents(docs: List[Document]) -> List[str]:
    """ドキュメントリストをテキストチャンクに分割"""
    if not docs:
        return []
    full_text = "\n".join(doc.page_content for doc in docs)
    chunks = SPLITTER.split_text(full_text)
    logger.info(f"Split into {len(chunks)} chunks")
    return chunks

# ── Candidate Generation Function ─────────────────
def generate_candidates_from_chunk(text: str) -> List[str]:
    """SudachiPyとN-gramでチャンクから候補語を生成する"""
    if not text.strip():
        return []
    
    try:
        # 各呼び出しごとに新しいtokenizerインスタンスを作成
        local_tokenizer = dictionary.Dictionary().create()
        
        # Mode.Aで分かち書き
        tokens = local_tokenizer.tokenize(text, sudachi_mode)
        
        # 名詞のみを抽出（位置情報も保持）
        noun_tokens = []
        for i, token in enumerate(tokens):
            pos = token.part_of_speech()[0]
            if pos == '名詞':
                noun_tokens.append({
                    'surface': token.surface(),
                    'position': i,
                    'normalized': token.normalized_form()
                })
        
        if not noun_tokens:
            return []
        
        candidates = set()
        
        # 単体の名詞（ユニグラム）を追加
        for token in noun_tokens:
            surface = token['surface']
            # 1文字の名詞は除外（ノイズ削減）
            if len(surface) > 1:
                candidates.add(surface)
        
        # 連続する名詞をグループ化
        noun_groups = []
        current_group = []
        
        for i, token in enumerate(noun_tokens):
            if not current_group:
                current_group.append(token)
            else:
                # 前のトークンと連続しているかチェック
                if token['position'] == current_group[-1]['position'] + 1:
                    current_group.append(token)
                else:
                    if len(current_group) >= 2:
                        noun_groups.append(current_group)
                    current_group = [token]
        
        # 最後のグループを追加
        if len(current_group) >= 2:
            noun_groups.append(current_group)
        
        # 各グループから2-gram～4-gramを生成
        for group in noun_groups:
            surfaces = [t['surface'] for t in group]
            for n in range(2, min(5, len(surfaces) + 1)):
                for i in range(len(surfaces) - n + 1):
                    candidate = "".join(surfaces[i:i+n])
                    candidates.add(candidate)
        
        # 候補リストを長さの降順でソート（長い複合語を優先）
        sorted_candidates = sorted(list(candidates), key=len, reverse=True)
        
        # 上位100件に制限（コンテキストウィンドウとコスト対策）
        return sorted_candidates[:100]
        
    except Exception as e:
        logger.error(f"Error in candidate generation: {e}")
        return []

# ── LCEL Chains with Tracing ─────────────────────

# Chain 1: ファイルからドキュメントをロード
load_document_chain = RunnableLambda(
    load_document,
    name="load_document"
)

# Chain 2: ドキュメントをチャンクに分割
split_documents_chain = RunnableLambda(
    split_documents,
    name="split_documents"  
)

# ファイル処理パイプライン（名前付き）
file_processing_pipeline = (
    load_document_chain
    | split_documents_chain
).with_config({"run_name": "file_processing"})

# 候補語生成チェイン（名前付き）
candidate_generation_chain = RunnableLambda(
    generate_candidates_from_chunk,
    name="generate_candidates"
)

# Chain 3: 新しい用語抽出チェイン（候補生成＋検証＋RAG）
async def extract_with_context(chunk_data: Dict[str, str]) -> Dict:
    """RAGを含む用語抽出"""
    chunk_text = chunk_data["text"]
    chunk_id = chunk_data["chunk_id"]
    
    # 候補語生成（トレース対応）
    candidates = await candidate_generation_chain.ainvoke(chunk_text)
    
    # 類似チャンクを検索
    related_contexts = await vector_store.search_similar_chunks(
        chunk_text[:1000],  # 検索用に最初の1000文字を使用
        chunk_id,
        n_results=3
    )
    
    # プロンプトに渡すデータを準備
    prompt_data = {
        "text": chunk_text[:3000],
        "candidates": "\n".join(candidates),
        "related_contexts": related_contexts
    }
    
    # LLMで用語抽出（名前付きチェイン）
    extraction_chain = (
        validation_prompt 
        | llm 
        | json_parser
    ).with_config({"run_name": "term_validation"})
    
    result = await extraction_chain.ainvoke(prompt_data)
    return result

# 用語抽出関数をチェイン化
extract_with_context_chain = RunnableLambda(
    extract_with_context,
    name="extract_with_rag"
)

# Chain 4: 用語統合チェイン（名前付き）
term_consolidation_chain = (
    RunnablePassthrough.assign(
        terms_json=lambda x: json.dumps(
            {"terms": x["terms"]}, 
            ensure_ascii=False
        )
    )
    | consolidate_prompt
    | llm
    | json_parser
).with_config({"run_name": "term_consolidation"})

# Chain 5: バッチ処理用の統合チェイン
async def consolidate_in_batches(all_terms: List[Dict]) -> List[Dict]:
    """大量の用語をバッチ処理で統合"""
    if len(all_terms) <= 50:
        result = await term_consolidation_chain.ainvoke({"terms": all_terms})
        return result.get("terms", [])
    
    # 30件ずつバッチ処理
    batch_size = 30
    consolidated = []
    
    for i in range(0, len(all_terms), batch_size):
        batch = all_terms[i:i+batch_size]
        result = await term_consolidation_chain.ainvoke({"terms": batch})
        consolidated.extend(result.get("terms", []))
        
        # レート制限対応
        if i + batch_size < len(all_terms):
            await asyncio.sleep(7)
    
    return consolidated

# ファイル処理パイプライン（定義済み）への参照を削除（上で定義済み）

# 関数定義（元の実装を維持）
def load_document(file_path: Path) -> List[Document]:
    """ファイルパスからドキュメントをロード"""
    try:
        loader_cls = LOADER_MAP.get(file_path.suffix.lower(), TextLoader)
        logger.info(f"Loading {file_path.name} with {loader_cls.__name__}")
        return loader_cls(str(file_path)).load()
    except Exception as e:
        logger.error(f"Failed to load {file_path}: {e}")
        return []

def split_documents(docs: List[Document]) -> List[str]:
    """ドキュメントリストをテキストチャンクに分割"""
    if not docs:
        return []
    full_text = "\n".join(doc.page_content for doc in docs)
    chunks = SPLITTER.split_text(full_text)
    logger.info(f"Split into {len(chunks)} chunks")
    return chunks

# チャンクからの用語抽出パイプライン（バッチ処理付き）
async def extract_terms_with_rate_limit(chunks_with_ids: List[Dict[str, str]]) -> List[TermList]:
    """レート制限を考慮した用語抽出（RAG対応）"""
    batch_size = 3
    delay_between_batches = 7
    
    results = []
    for i in range(0, len(chunks_with_ids), batch_size):
        batch = chunks_with_ids[i:i+batch_size]
        
        # バッチ内の並列処理（トレース対応）
        batch_results = await asyncio.gather(
            *(extract_with_context_chain.ainvoke(chunk_data) for chunk_data in batch)
        )
        results.extend(batch_results)
        
        # レート制限対応
        if i + batch_size < len(chunks_with_ids):
            logger.info(f"Processed {i + len(batch)}/{len(chunks_with_ids)} chunks. Waiting {delay_between_batches}s...")
            await asyncio.sleep(delay_between_batches)
    
    return results

# 重複削除パイプライン
def merge_duplicate_terms(term_lists: List[TermList]) -> List[Term]:
    """重複する用語をマージ"""
    merged: Dict[str, Dict] = {}
    
    for term_list in term_lists:
        for term in term_list.get("terms", []):
            if not isinstance(term, dict) or not term.get("headword"):
                continue
                
            key = term["headword"].lower().strip()
            if key not in merged:
                # デフォルト値を確保
                merged[key] = {
                    "headword": term.get("headword", ""),
                    "synonyms": term.get("synonyms", []),
                    "definition": term.get("definition", ""),
                    "category": term.get("category", None)
                }
            else:
                # 類義語をマージ
                existing_synonyms = set(merged[key].get("synonyms", []))
                new_synonyms = set(term.get("synonyms", []))
                merged[key]["synonyms"] = list(existing_synonyms | new_synonyms)
                
                # 定義が空の場合は更新
                if not merged[key].get("definition") and term.get("definition"):
                    merged[key]["definition"] = term["definition"]
                
                # カテゴリが空の場合は更新
                if not merged[key].get("category") and term.get("category"):
                    merged[key]["category"] = term["category"]
    
    logger.info(f"Merged {len(merged)} unique terms")
    return list(merged.values())

# メインパイプライン
async def run_pipeline(input_dir: Path, output_json: Path):
    """メインの処理パイプライン（RAG対応版）"""
    
    # 1. ファイル検索
    files = []
    for ext in LOADER_MAP:
        files.extend(input_dir.glob(f"**/*{ext}"))
    
    if not files:
        logger.error(f"No supported files found in {input_dir}")
        return
    
    logger.info(f"Found {len(files)} files to process")
    
    # 2. ファイル処理パイプライン（並列実行）
    file_chunks = await asyncio.gather(
        *(file_processing_pipeline.ainvoke(f) for f in files)
    )
    
    # 3. 全チャンクをフラット化
    all_chunks = [chunk for chunks in file_chunks for chunk in chunks if chunk.strip()]
    
    if not all_chunks:
        logger.error("No text chunks generated")
        return
    
    logger.info(f"Total chunks to process: {len(all_chunks)}")
    
    # 4. チャンクIDを生成
    chunk_ids = [f"chunk_{i}" for i in range(len(all_chunks))]
    chunks_with_ids = [
        {"text": chunk, "chunk_id": chunk_id} 
        for chunk, chunk_id in zip(all_chunks, chunk_ids)
    ]
    
    # 5. ベクトルストアを初期化（全チャンクをエンベディング化）
    logger.info("Initializing vector store...")
    await vector_store.initialize(all_chunks, chunk_ids)
    
    # 6. 用語抽出（レート制限付き、RAG使用）
    term_lists = await extract_terms_with_rate_limit(chunks_with_ids)
    
    # 7. 空の結果を除外
    valid_term_lists = [tl for tl in term_lists if tl.get("terms")]
    logger.info(f"Chunks with terms: {len(valid_term_lists)}")
    
    if not valid_term_lists:
        logger.error("No terms extracted from any chunk")
        final_terms = []
    else:
        # 8. 重複削除
        unique_terms = merge_duplicate_terms(valid_term_lists)
        
        # 9. 最終統合
        final_terms = await consolidate_in_batches(unique_terms)
    
    # 10. 保存
    output_data = {"terms": final_terms}
    
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Saved {len(final_terms)} terms to {output_json}")
    
    # デバッグ用：最初の3件を表示
    if final_terms:
        logger.info("Sample terms:")
        for term in final_terms[:3]:
            headword = term.get("headword", "N/A")
            definition = term.get("definition", "N/A")
            logger.info(f"  - {headword}: {definition[:50]}...")

# ── Entry Point ───────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("Usage: python term_extractor_lcel.py <input_dir> <output_json>")
    
    asyncio.run(run_pipeline(Path(sys.argv[1]), Path(sys.argv[2])))