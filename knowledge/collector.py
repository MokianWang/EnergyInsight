"""
文档采集器
支持从URL下载和本地目录扫描两种方式
"""

import re
import httpx
from pathlib import Path


def collect_from_urls(urls: list[str]) -> list[dict]:
    """
    从URL下载文档，支持PDF和HTML

    Args:
        urls: 文档URL列表

    Returns:
        文档元数据列表
    """
    downloads = []
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
    }

    for url in urls:
        try:
            with httpx.Client(timeout=60, follow_redirects=True) as client:
                resp = client.get(url, headers=headers)
                resp.raise_for_status()
                content = resp.content
                content_type = resp.headers.get("content-type", "")

            # 确定文件类型
            if "pdf" in content_type or url.lower().endswith(".pdf"):
                ext = ".pdf"
                file_type = "pdf"
            else:
                ext = ".html"
                file_type = "html"

            # 生成文件名
            filename = _url_to_filename(url) + ext
            save_path = os.path.join("data/raw", filename)

            # 保存
            with open(save_path, "wb") as f:
                f.write(content)

            title = filename.replace(ext, "").replace("_", " ")
            downloads.append({
                "source": "url",
                "source_url": url,
                "local_path": os.path.abspath(save_path),
                "title": title,
                "file_type": file_type,
            })
            print(f"[Collector] 已下载: {url[:60]}... -> {filename}")

        except Exception as e:
            print(f"[Collector] 下载失败 {url[:60]}...: {e}")

    return downloads


def collect_from_local(paths: list[str]) -> list[dict]:
    """
    扫描本地路径，收集PDF/TXT/MD文件

    Args:
        paths: 文件路径或目录路径列表

    Returns:
        文档元数据列表
    """
    docs = []
    supported = {".pdf", ".txt", ".md", ".html", ".htm"}

    for path in paths:
        p = Path(path)
        if p.is_file():
            if p.suffix.lower() in supported:
                docs.append(_file_to_doc(p))
        elif p.is_dir():
            for f in p.rglob("*"):
                if f.is_file() and f.suffix.lower() in supported:
                    docs.append(_file_to_doc(f))
        else:
            print(f"[Collector] 路径不存在: {path}")

    print(f"[Collector] 本地扫描: {len(docs)} 个文档")
    return docs


def list_local_files(directory: str, extensions: str = ".pdf,.txt,.md") -> list[str]:
    """
    列出目录下所有指定类型的文件

    Args:
        directory: 目录路径
        extensions: 逗号分隔的文件扩展名

    Returns:
        文件路径列表
    """
    exts = set(extensions.split(","))
    files = []
    for f in Path(directory).rglob("*"):
        if f.is_file() and f.suffix.lower() in exts:
            files.append(str(f))
    return sorted(files)


def _file_to_doc(filepath: Path) -> dict:
    ext = filepath.suffix.lower()
    type_map = {".pdf": "pdf", ".txt": "txt", ".md": "md", ".html": "html", ".htm": "html"}
    return {
        "source": "local",
        "source_url": "",
        "local_path": str(filepath.absolute()),
        "title": filepath.stem,
        "file_type": type_map.get(ext, "unknown"),
    }


def _url_to_filename(url: str) -> str:
    """URL 转安全的文件名"""
    name = re.sub(r"https?://", "", url)
    name = re.sub(r"[?#&=:/\\]", "_", name)
    return name[:80]
