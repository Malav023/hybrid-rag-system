import logging
import os
import uuid
from typing import Dict

from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, UploadFile, status

from api.dependencies import get_dense_index, get_sparse_index, verify_api_key
from config.settings import settings
from ingestion import IngestionPipeline

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ingest", tags=["Ingestion"])

# In-memory job store — swap for Redis/SQLite in production
_jobs: Dict[str, dict] = {}

_ALLOWED_EXTENSIONS = {
    ".pdf", ".docx", ".xlsx", ".xls",
    ".csv", ".pptx", ".ppt", ".html", ".htm", ".txt", ".md",
}

_MAX_FILE_SIZE_BYTES = 100 * 1024 * 1024


def _run_ingestion(job_id: str, save_path: str, filename: str,
                   dense_index, sparse_index) -> None:
    _jobs[job_id]["status"] = "processing"
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
        _jobs[job_id]["status"] = "done"
        _jobs[job_id]["result"] = {
            "source_file": filename,
            "chunks_indexed": len(chunks),
            "message": f"Successfully indexed {len(chunks)} chunks from '{filename}'.",
        }
        logger.info(f"Job {job_id} complete — {len(chunks)} chunks")
    except Exception as exc:
        logger.exception(f"Job {job_id} failed")
        _jobs[job_id]["status"] = "error"
        _jobs[job_id]["error"] = str(exc)


@router.post(
    "/",
    status_code=status.HTTP_202_ACCEPTED,
    summary="Upload and ingest a document (async)",
    dependencies=[Depends(verify_api_key)],
)
async def ingest_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    dense_index=Depends(get_dense_index),
    sparse_index=Depends(get_sparse_index),
):
    filename = file.filename or ""
    ext = os.path.splitext(filename)[-1].lower()
    if ext not in _ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"File type '{ext}' is not supported. Accepted: {sorted(_ALLOWED_EXTENSIONS)}",
        )

    upload_dir = str(settings.UPLOAD_DIR)
    os.makedirs(upload_dir, exist_ok=True)
    save_path = os.path.join(upload_dir, filename)

    try:
        total_bytes = 0
        with open(save_path, "wb") as out:
            while chunk := await file.read(1024 * 256):
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

    job_id = str(uuid.uuid4())
    _jobs[job_id] = {"status": "queued", "result": None, "error": None}

    background_tasks.add_task(
        _run_ingestion, job_id, save_path, filename, dense_index, sparse_index
    )

    return {"job_id": job_id, "status": "queued", "filename": filename}


@router.get(
    "/status/{job_id}",
    summary="Poll ingestion job status",
    dependencies=[Depends(verify_api_key)],
)
def ingest_status(job_id: str):
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")

    response = {"status": job["status"]}
    if job["status"] == "done":
        response["result"] = job["result"]
    elif job["status"] == "error":
        response["error"] = job["error"]
    return response