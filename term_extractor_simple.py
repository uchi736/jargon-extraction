#!/usr/bin/env python3
"""term_extractor_lcel.py
LCEL記法版 専門用語・類義語辞書生成
------------------------------------------------
* LangChain Expression Language (LCEL) でチェイン構築
* 構造化出力 (Pydantic) で型安全性確保
* RunnablePassthrough/RunnableLambda で処理フロー明確化
* `.env` から `GOOGLE_API_KEY` 読み込み

Quick Start::
    python -m venv .venv && .venv/Scripts/activate   # ← Windows
    pip install -U langchain langchain-google-genai \
                 langchain-core pydantic python-dotenv \
                 unstructured[all] pypdf python-docx

    echo GOOGLE_API_KEY=YOUR_KEY > .env
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

# ── ENV ───────────────────────────────────────────
load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    sys.exit("[ERROR] .env に GOOGLE_API_KEY を設定してください")

# ── LangChain imports ─────────────────────────────
from langchain_google_genai import ChatGoogleGenerativeAI
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

# ── Output Parser ─────────────────────────────────
json_parser = JsonOutputParser(pydantic_object=TermList)

# ── Prompts ───────────────────────────────────────
extract_prompt = ChatPromptTemplate.from_messages([
    ("system", """あなたは技術文書から専門用語を抽出する専門家です。
必ず以下の形式の有効なJSONのみを返してください。他の説明は一切不要です。

{format_instructions}

一般的すぎる単語（例：システム、データ、情報）は除外し、その分野特有の専門用語のみを抽出してください。"""),
    ("user", "以下のテキストから専門用語を抽出してJSON形式で返してください:\n\n{text}"),
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

# ── LCEL Chains ───────────────────────────────────

# Chain 1: ファイルからドキュメントをロード
def load_document(file_path: Path) -> List[Document]:
    """ファイルパスからドキュメントをロード"""
    try:
        loader_cls = LOADER_MAP.get(file_path.suffix.lower(), TextLoader)
        logger.info(f"Loading {file_path.name} with {loader_cls.__name__}")
        return loader_cls(str(file_path)).load()
    except Exception as e:
        logger.error(f"Failed to load {file_path}: {e}")
        return []

# Chain 2: ドキュメントをチャンクに分割
def split_documents(docs: List[Document]) -> List[str]:
    """ドキュメントリストをテキストチャンクに分割"""
    if not docs:
        return []
    full_text = "\n".join(doc.page_content for doc in docs)
    chunks = SPLITTER.split_text(full_text)
    logger.info(f"Split into {len(chunks)} chunks")
    return chunks

# Chain 3: 用語抽出チェイン
term_extraction_chain = (
    RunnablePassthrough.assign(
        text=lambda x: x["text"][:3000]  # 長すぎる場合は切る
    )
    | extract_prompt
    | llm
    | json_parser
)

# Chain 4: 用語統合チェイン
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
)

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

# ── Main Pipeline using LCEL ─────────────────────

# ファイル処理パイプライン
file_processing_pipeline = (
    RunnableLambda(load_document)
    | RunnableLambda(split_documents)
)

# チャンクからの用語抽出パイプライン（バッチ処理付き）
async def extract_terms_with_rate_limit(chunks: List[str]) -> List[TermList]:
    """レート制限を考慮した用語抽出"""
    batch_size = 3
    delay_between_batches = 7
    
    results = []
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i+batch_size]
        
        # バッチ内の並列処理
        batch_results = await asyncio.gather(
            *(term_extraction_chain.ainvoke({"text": chunk}) for chunk in batch)
        )
        results.extend(batch_results)
        
        # レート制限対応
        if i + batch_size < len(chunks):
            logger.info(f"Processed {i + len(batch)}/{len(chunks)} chunks. Waiting {delay_between_batches}s...")
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
    """メインの処理パイプライン"""
    
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
    
    # 4. 用語抽出（レート制限付き）
    term_lists = await extract_terms_with_rate_limit(all_chunks)
    
    # 5. 空の結果を除外
    valid_term_lists = [tl for tl in term_lists if tl["terms"]]
    logger.info(f"Chunks with terms: {len(valid_term_lists)}")
    
    if not valid_term_lists:
        logger.error("No terms extracted from any chunk")
        final_terms = []
    else:
        # 6. 重複削除
        unique_terms = merge_duplicate_terms(valid_term_lists)
        
        # 7. 最終統合
        final_terms = await consolidate_in_batches(unique_terms)
    
    # 8. 保存
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