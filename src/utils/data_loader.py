"""
Tiện ích để tải và xử lý dữ liệu cho RAG pipeline.

Cách dùng:
    from utils.data_loader import load_knowledge_base, split_text, build_vectorstore

    text        = load_knowledge_base()
    chunks      = split_text(text, chunk_size=500, chunk_overlap=50)
    vectorstore = build_vectorstore(chunks, embeddings)
"""
from pathlib import Path


def load_knowledge_base(path: str = None) -> str:
    """
    Đọc file knowledge base và trả về nội dung dạng chuỗi.

    Args:
        path: đường dẫn tới file text.
              Mặc định: data/knowledge_base.txt (thư mục gốc của project)

    Returns:
        Nội dung file dưới dạng str
    """
    if path is None:
        path = Path(__file__).parent.parent.parent / "data" / "knowledge_base.txt"
    return Path(path).read_text(encoding="utf-8")


def split_text(text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> list:
    """
    Chia văn bản thành các đoạn nhỏ (chunks) để index.

    Dùng RecursiveCharacterTextSplitter — tách ưu tiên theo đoạn văn, câu, rồi ký tự.

    Args:
        text         : văn bản cần chia
        chunk_size   : số ký tự tối đa mỗi chunk (mặc định: 500)
        chunk_overlap: số ký tự chồng lên nhau giữa 2 chunks liên tiếp (mặc định: 50)

    Returns:
        list[str] — danh sách các chuỗi chunk
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return splitter.split_text(text)


def build_vectorstore(chunks: list, embeddings, cache_dir: str = None):
    """
    Tạo FAISS vectorstore từ danh sách chunks và embeddings.
    Tự động batching và xử lý rate limit (429) cho các API miễn phí (Gemini),
    đồng thời lưu cache để không phải embed lại ở các lần chạy sau.

    Args:
        chunks    : list[str] — danh sách text chunks đã chia
        embeddings: Embeddings instance (từ get_embeddings())
        cache_dir : Thư mục lưu cache FAISS index

    Returns:
        FAISS vectorstore đã được index và sẵn sàng dùng để retrieve
    """
    import time
    from langchain_community.vectorstores import FAISS

    if cache_dir is None:
        cache_dir = Path(__file__).parent.parent.parent / "data" / "faiss_index"
    else:
        cache_dir = Path(cache_dir)

    # 1. Nếu đã có cache từ lần chạy trước -> load ngay lập tức
    index_file = cache_dir / "index.faiss"
    if index_file.exists():
        try:
            print(f"📦 Tìm thấy FAISS index đã lưu tại {cache_dir}, đang tải...")
            vectorstore = FAISS.load_local(str(cache_dir), embeddings, allow_dangerous_deserialization=True)
            print(f"✅ Đã tải thành công FAISS vectorstore ({vectorstore.index.ntotal} vectors).")
            return vectorstore
        except Exception as e:
            print(f"⚠️ Không thể nạp cache: {e}. Tiến hành build lại...")

    # 2. Build index mới theo từng batch (50 chunks/batch) để tránh rate-limit 100 req/min
    print(f"🔨 Đang tạo FAISS index từ {len(chunks)} chunks ...")
    batch_size = 50
    vectorstore = None

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        batch_num = (i // batch_size) + 1
        total_batches = (len(chunks) + batch_size - 1) // batch_size
        print(f"   Đang xử lý batch {batch_num}/{total_batches} ({len(batch)} chunks)...")

        # Retry logic nếu gặp 429 RESOURCE_EXHAUSTED
        for attempt in range(5):
            try:
                vs_part = FAISS.from_texts(batch, embeddings)
                if vectorstore is None:
                    vectorstore = vs_part
                else:
                    vectorstore.merge_from(vs_part)
                break
            except Exception as e:
                err_str = str(e)
                if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                    wait_sec = 35
                    print(f"   ⏳ Gặp rate limit (429). Đang chờ {wait_sec}s để reset quota rồi thử lại...")
                    time.sleep(wait_sec)
                else:
                    raise e

    # 3. Lưu vào cache để các script sau dùng lại ngay lập tức
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        vectorstore.save_local(str(cache_dir))
        print(f"💾 Đã lưu cache FAISS index tại {cache_dir}")
    except Exception as e:
        print(f"⚠️ Không thể lưu cache: {e}")

    print("✅ FAISS vectorstore đã sẵn sàng.")
    return vectorstore

