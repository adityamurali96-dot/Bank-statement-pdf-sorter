"""
PDF to JSON Converter API using Docling
Deployable on Railway with comprehensive error handling and debugging
"""

import io
import json
import logging
import os
import sys
import tempfile
import traceback
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

# Configure logging before any other imports
logging.basicConfig(
    level=logging.DEBUG if os.getenv("DEBUG", "false").lower() == "true" else logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)

logger = logging.getLogger(__name__)

# Import Docling components with error handling
try:
    from docling.document_converter import DocumentConverter
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.document_converter import PdfFormatOption
    DOCLING_AVAILABLE = True
    logger.info("Docling successfully imported")
except ImportError as e:
    DOCLING_AVAILABLE = False
    DOCLING_IMPORT_ERROR = str(e)
    logger.error(f"Failed to import Docling: {e}")
    logger.error(f"Traceback: {traceback.format_exc()}")


# === Response Models ===

class ErrorDetail(BaseModel):
    """Detailed error information for debugging"""
    error_code: str
    message: str
    details: Optional[str] = None
    timestamp: str
    request_id: str
    traceback: Optional[str] = None


class ConversionResponse(BaseModel):
    """Successful conversion response"""
    success: bool
    request_id: str
    filename: str
    conversion_time_seconds: float
    document: Dict[str, Any]
    metadata: Dict[str, Any]


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    docling_available: bool
    version: str
    timestamp: str
    environment: Dict[str, str]


# === Application Lifespan ===

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler for startup/shutdown events"""
    logger.info("=" * 60)
    logger.info("PDF to JSON Converter API Starting...")
    logger.info("=" * 60)

    # Log environment info
    logger.info(f"Python version: {sys.version}")
    logger.info(f"Working directory: {os.getcwd()}")
    logger.info(f"DEBUG mode: {os.getenv('DEBUG', 'false')}")
    logger.info(f"Docling available: {DOCLING_AVAILABLE}")

    if not DOCLING_AVAILABLE:
        logger.error(f"Docling import error: {DOCLING_IMPORT_ERROR}")

    # Initialize converter if Docling is available
    if DOCLING_AVAILABLE:
        try:
            logger.info("Initializing Docling DocumentConverter...")
            app.state.converter = create_document_converter()
            logger.info("DocumentConverter initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize DocumentConverter: {e}")
            logger.error(traceback.format_exc())
            app.state.converter = None
    else:
        app.state.converter = None

    logger.info("=" * 60)
    logger.info("API Ready to accept requests")
    logger.info("=" * 60)

    yield

    # Shutdown
    logger.info("API Shutting down...")


# === Helper Functions ===

def create_document_converter() -> "DocumentConverter":
    """Create and configure the Docling DocumentConverter"""
    logger.debug("Creating DocumentConverter with PDF pipeline options")

    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = True  # Enable OCR for scanned documents
    pipeline_options.do_table_structure = True  # Extract table structures

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pipeline_options
            )
        }
    )

    return converter


def generate_request_id() -> str:
    """Generate a unique request ID for tracking"""
    return f"req_{uuid.uuid4().hex[:12]}"


def get_error_response(
    error_code: str,
    message: str,
    request_id: str,
    status_code: int = 500,
    details: Optional[str] = None,
    include_traceback: bool = False,
) -> JSONResponse:
    """Generate a standardized error response"""
    error_detail = ErrorDetail(
        error_code=error_code,
        message=message,
        details=details,
        timestamp=datetime.utcnow().isoformat() + "Z",
        request_id=request_id,
        traceback=traceback.format_exc() if include_traceback else None,
    )

    logger.error(f"[{request_id}] Error {error_code}: {message}")
    if details:
        logger.error(f"[{request_id}] Details: {details}")
    if include_traceback:
        logger.error(f"[{request_id}] Traceback:\n{traceback.format_exc()}")

    return JSONResponse(
        status_code=status_code,
        content={"success": False, "error": error_detail.model_dump()},
    )


# === FastAPI Application ===

app = FastAPI(
    title="PDF to JSON Converter API",
    description="Convert PDF documents to structured JSON using Docling",
    version="1.0.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure templates
templates_dir = Path(__file__).parent / "templates"
templates = Jinja2Templates(directory=str(templates_dir))


# === Exception Handlers ===

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler for uncaught errors"""
    request_id = getattr(request.state, "request_id", generate_request_id())

    logger.exception(f"[{request_id}] Unhandled exception: {exc}")

    return get_error_response(
        error_code="INTERNAL_SERVER_ERROR",
        message="An unexpected error occurred",
        request_id=request_id,
        status_code=500,
        details=str(exc),
        include_traceback=os.getenv("DEBUG", "false").lower() == "true",
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handler for HTTP exceptions"""
    request_id = getattr(request.state, "request_id", generate_request_id())

    logger.warning(f"[{request_id}] HTTP exception: {exc.status_code} - {exc.detail}")

    return get_error_response(
        error_code=f"HTTP_{exc.status_code}",
        message=str(exc.detail),
        request_id=request_id,
        status_code=exc.status_code,
    )


# === Middleware ===

@app.middleware("http")
async def request_logging_middleware(request: Request, call_next):
    """Log all incoming requests and add request ID"""
    request_id = generate_request_id()
    request.state.request_id = request_id

    logger.info(f"[{request_id}] {request.method} {request.url.path} - Start")
    logger.debug(f"[{request_id}] Headers: {dict(request.headers)}")

    start_time = datetime.utcnow()

    try:
        response = await call_next(request)
        duration = (datetime.utcnow() - start_time).total_seconds()

        logger.info(
            f"[{request_id}] {request.method} {request.url.path} - "
            f"Status: {response.status_code} - Duration: {duration:.3f}s"
        )

        response.headers["X-Request-ID"] = request_id
        return response

    except Exception as e:
        duration = (datetime.utcnow() - start_time).total_seconds()
        logger.error(
            f"[{request_id}] {request.method} {request.url.path} - "
            f"Error after {duration:.3f}s: {e}"
        )
        raise


# === API Endpoints ===

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    """Serve the upload UI"""
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/api", response_model=Dict[str, str])
async def api_info():
    """API information endpoint"""
    return {
        "message": "PDF to JSON Converter API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health", response_model=HealthResponse)
async def health_check(request: Request):
    """Health check endpoint with system information"""
    request_id = getattr(request.state, "request_id", generate_request_id())

    logger.debug(f"[{request_id}] Health check requested")

    docling_status = "available" if DOCLING_AVAILABLE else "unavailable"
    converter_status = "initialized" if (hasattr(app.state, "converter") and app.state.converter) else "not_initialized"

    return HealthResponse(
        status="healthy" if DOCLING_AVAILABLE else "degraded",
        docling_available=DOCLING_AVAILABLE,
        version="1.0.0",
        timestamp=datetime.utcnow().isoformat() + "Z",
        environment={
            "python_version": sys.version.split()[0],
            "docling_status": docling_status,
            "converter_status": converter_status,
            "debug_mode": os.getenv("DEBUG", "false"),
        },
    )


@app.post("/convert")
async def convert_pdf_to_json(
    request: Request,
    file: UploadFile = File(..., description="PDF file to convert"),
):
    """
    Convert a PDF file to structured JSON

    This endpoint accepts a PDF file and returns the extracted content as JSON,
    including text, tables, and document structure.

    **Error Codes:**
    - DOCLING_UNAVAILABLE: Docling library is not available
    - CONVERTER_NOT_INITIALIZED: Document converter failed to initialize
    - INVALID_FILE_TYPE: Uploaded file is not a PDF
    - FILE_READ_ERROR: Failed to read uploaded file
    - CONVERSION_ERROR: Failed to convert PDF to JSON
    - SERIALIZATION_ERROR: Failed to serialize document to JSON
    """
    request_id = getattr(request.state, "request_id", generate_request_id())
    start_time = datetime.utcnow()

    logger.info(f"[{request_id}] PDF conversion requested")
    logger.info(f"[{request_id}] Filename: {file.filename}")
    logger.info(f"[{request_id}] Content-Type: {file.content_type}")

    # Check if Docling is available
    if not DOCLING_AVAILABLE:
        return get_error_response(
            error_code="DOCLING_UNAVAILABLE",
            message="Docling library is not available",
            request_id=request_id,
            status_code=503,
            details=f"Import error: {DOCLING_IMPORT_ERROR}",
        )

    # Check if converter is initialized
    if not hasattr(app.state, "converter") or app.state.converter is None:
        return get_error_response(
            error_code="CONVERTER_NOT_INITIALIZED",
            message="Document converter is not initialized",
            request_id=request_id,
            status_code=503,
            details="The converter failed to initialize at startup. Check logs for details.",
        )

    # Validate file type
    if not file.filename:
        return get_error_response(
            error_code="INVALID_FILE_TYPE",
            message="No filename provided",
            request_id=request_id,
            status_code=400,
        )

    if not file.filename.lower().endswith(".pdf"):
        return get_error_response(
            error_code="INVALID_FILE_TYPE",
            message="Only PDF files are accepted",
            request_id=request_id,
            status_code=400,
            details=f"Received file: {file.filename}",
        )

    # Read file content
    try:
        logger.debug(f"[{request_id}] Reading file content...")
        file_content = await file.read()
        file_size = len(file_content)
        logger.info(f"[{request_id}] File size: {file_size} bytes")

        if file_size == 0:
            return get_error_response(
                error_code="FILE_READ_ERROR",
                message="Uploaded file is empty",
                request_id=request_id,
                status_code=400,
            )

    except Exception as e:
        return get_error_response(
            error_code="FILE_READ_ERROR",
            message="Failed to read uploaded file",
            request_id=request_id,
            status_code=400,
            details=str(e),
            include_traceback=True,
        )

    # Convert PDF using Docling
    temp_file_path = None
    try:
        # Save to temporary file (Docling requires file path)
        logger.debug(f"[{request_id}] Creating temporary file...")
        with tempfile.NamedTemporaryFile(
            suffix=".pdf",
            delete=False,
            prefix=f"pdf_convert_{request_id}_"
        ) as temp_file:
            temp_file.write(file_content)
            temp_file_path = temp_file.name

        logger.info(f"[{request_id}] Temporary file created: {temp_file_path}")

        # Perform conversion
        logger.info(f"[{request_id}] Starting Docling conversion...")
        conversion_start = datetime.utcnow()

        result = app.state.converter.convert(temp_file_path)

        conversion_duration = (datetime.utcnow() - conversion_start).total_seconds()
        logger.info(f"[{request_id}] Docling conversion completed in {conversion_duration:.3f}s")

    except Exception as e:
        logger.error(f"[{request_id}] Conversion failed: {e}")
        return get_error_response(
            error_code="CONVERSION_ERROR",
            message="Failed to convert PDF",
            request_id=request_id,
            status_code=500,
            details=str(e),
            include_traceback=True,
        )
    finally:
        # Clean up temporary file
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
                logger.debug(f"[{request_id}] Temporary file cleaned up")
            except Exception as e:
                logger.warning(f"[{request_id}] Failed to clean up temp file: {e}")

    # Serialize result to JSON
    try:
        logger.debug(f"[{request_id}] Serializing document to JSON...")

        # Export document to dictionary format
        document_dict = result.document.export_to_dict()

        # Also get markdown representation for convenience
        markdown_content = result.document.export_to_markdown()

        # Build response
        total_duration = (datetime.utcnow() - start_time).total_seconds()

        response_data = {
            "success": True,
            "request_id": request_id,
            "filename": file.filename,
            "conversion_time_seconds": round(total_duration, 3),
            "document": document_dict,
            "metadata": {
                "file_size_bytes": file_size,
                "conversion_timestamp": datetime.utcnow().isoformat() + "Z",
                "markdown_preview": markdown_content[:1000] if markdown_content else None,
                "has_tables": bool(document_dict.get("tables")),
                "has_pictures": bool(document_dict.get("pictures")),
                "page_count": len(document_dict.get("pages", [])),
            },
        }

        logger.info(f"[{request_id}] Conversion successful - Total time: {total_duration:.3f}s")
        logger.info(f"[{request_id}] Pages: {response_data['metadata']['page_count']}")

        return JSONResponse(content=response_data)

    except Exception as e:
        logger.error(f"[{request_id}] Serialization failed: {e}")
        return get_error_response(
            error_code="SERIALIZATION_ERROR",
            message="Failed to serialize document to JSON",
            request_id=request_id,
            status_code=500,
            details=str(e),
            include_traceback=True,
        )


@app.post("/convert/markdown")
async def convert_pdf_to_markdown(
    request: Request,
    file: UploadFile = File(..., description="PDF file to convert"),
):
    """
    Convert a PDF file to Markdown format

    This endpoint is a simpler alternative that returns the document as markdown text.
    """
    request_id = getattr(request.state, "request_id", generate_request_id())

    logger.info(f"[{request_id}] Markdown conversion requested for: {file.filename}")

    # Reuse the main conversion logic
    json_response = await convert_pdf_to_json(request, file)

    # If there was an error, return it
    if isinstance(json_response, JSONResponse):
        response_body = json.loads(json_response.body.decode())
        if not response_body.get("success"):
            return json_response

        # Extract markdown from successful response
        return JSONResponse(content={
            "success": True,
            "request_id": request_id,
            "filename": file.filename,
            "markdown": response_body.get("metadata", {}).get("markdown_preview", ""),
            "full_markdown_in_document": True,
        })

    return json_response


@app.get("/debug/info")
async def debug_info(request: Request):
    """
    Debug endpoint showing system information
    Only available when DEBUG=true
    """
    request_id = getattr(request.state, "request_id", generate_request_id())

    if os.getenv("DEBUG", "false").lower() != "true":
        return get_error_response(
            error_code="DEBUG_DISABLED",
            message="Debug endpoint is disabled",
            request_id=request_id,
            status_code=403,
            details="Set DEBUG=true environment variable to enable",
        )

    return {
        "request_id": request_id,
        "system": {
            "python_version": sys.version,
            "platform": sys.platform,
            "cwd": os.getcwd(),
        },
        "environment": {k: v for k, v in os.environ.items() if not any(
            secret in k.lower() for secret in ["key", "secret", "password", "token"]
        )},
        "docling": {
            "available": DOCLING_AVAILABLE,
            "import_error": DOCLING_IMPORT_ERROR if not DOCLING_AVAILABLE else None,
        },
        "converter": {
            "initialized": hasattr(app.state, "converter") and app.state.converter is not None,
        },
    }


# === Main Entry Point ===

if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8000"))
    host = os.getenv("HOST", "0.0.0.0")

    logger.info(f"Starting server on {host}:{port}")

    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=os.getenv("DEBUG", "false").lower() == "true",
        log_level="debug" if os.getenv("DEBUG", "false").lower() == "true" else "info",
    )
