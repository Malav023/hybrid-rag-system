import logging
import os
import shutil

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status

from api.dependencies import get_dense_index, get_sparse_index, verify_api_key
from api.schemas import IngestResponse
from config.settings import settings
from ingestion import IngestionPipeline

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ingest", tags=["Ingestion"])

# Allowed MIME types → accepted extensions (defence-in-depth on top of ext check)
_ALLOWED_CONTENT_TYPES = {
    "application/pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "application/vnd.ms-excel",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    "text/csv",
    "text/plain",
    "text/html",
    "text/markdown",
    "application/octet-stream",   # some clients send this for binary files
}

_ALLOWED_EXTENSIONS = {
    ".pdf", ".docx", ".xlsx", ".xls",
    ".csv", ".pptx", ".ppt", ".html", ".htm", ".txt", ".md",
}

_MAX_FILE_SIZE_BYTES = 100 * 1024 * 1024   # 100 MB hard cap


@router.post(
    "/",
    response_model=IngestResponse,
    status_code=status.HTTP_200_OK,
    summary="Upload and ingest a document",
    dependencies=[Depends(verify_api_key)],
)
async def ingest_file(
    file: UploadFile = File(...),
    dense_index=Depends(get_dense_index),
    sparse_index=Depends(get_sparse_index),
):
    """
    Upload a document (PDF, DOCX, XLSX, CSV, PPTX, HTML, TXT, MD).

    The file is:
    1. Validated (extension + size)
    2. Saved to UPLOAD_DIR
    3. Parsed, chunked, and indexed into both dense and sparse stores
    4. De-duplicated automatically — re-uploading the same filename replaces
       its previous chunks in both indexes

    Returns chunk count and source filename on success.
    """
    # --- Extension validation ---
    filename = file.filename or ""
    ext = os.path.splitext(filename)[-1].lower()
    if ext not in _ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"File type '{ext}' is not supported. "
                   f"Accepted: {sorted(_ALLOWED_EXTENSIONS)}",
        )

    # --- Ensure upload dir exists ---
    upload_dir: str = str(settings.UPLOAD_DIR)
    os.makedirs(upload_dir, exist_ok=True)

    save_path = os.path.join(upload_dir, filename)

    # --- Stream file to disk with size enforcement ---
    try:
        total_bytes = 0
        with open(save_path, "wb") as out:
            while chunk := await file.read(1024 * 256):   # 256 KB chunks
                total_bytes += len(chunk)
                if total_bytes > _MAX_FILE_SIZE_BYTES:
                    out.close()
                    os.remove(save_path)
                    raise HTTPException(
                        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        detail=f"File exceeds the {_MAX_FILE_SIZE_BYTES // (1024*1024)} MB limit.",
                    )
                out.write(chunk)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception(f"Failed to save uploaded file '{filename}'")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to save file to disk.",
        ) from exc

    logger.info(f"Saved upload: {save_path} ({total_bytes / 1024:.1f} KB)")

    # --- Parse + chunk + index ---
    try:
        pipeline = IngestionPipeline(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            table_row_batch_size=settings.TABLE_ROW_BATCH_SIZE,
        )
        chunks = pipeline.ingest_file(
            file_path=save_path,
            dense_index=dense_index,
            sparse_index=sparse_index,
        )
    except ValueError as exc:
        # Unsupported extension slipped through (shouldn't happen, but belt+braces)
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        logger.exception(f"Ingestion failed for '{filename}'")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ingestion failed: {exc}",
        ) from exc

    logger.info(f"Ingestion complete: {filename} → {len(chunks)} chunks")

    return IngestResponse(
        source_file=filename,
        chunks_indexed=len(chunks),
        message=f"Successfully indexed {len(chunks)} chunks from '{filename}'.",
    )