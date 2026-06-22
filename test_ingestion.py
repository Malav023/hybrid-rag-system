# test_ingestion.py  — run from project root
import logging
logging.basicConfig(level=logging.INFO)

from ingestion import IngestionPipeline

pipeline = IngestionPipeline(chunk_size=500, chunk_overlap=50)

# Single file
chunks = pipeline.ingest_file("samples/report.pdf")
for c in chunks[:3]:
    print(c.metadata["chunk_type"], "|", c.text[:120])
    print("---")

# Whole directory
all_chunks = pipeline.ingest_directory("samples/")
print(f"\nTotal chunks: {len(all_chunks)}")