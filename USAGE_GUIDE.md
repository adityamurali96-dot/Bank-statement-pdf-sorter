# PDF to JSON Converter

A FastAPI-based web service that converts PDF documents to structured JSON using Docling. Designed for deployment on Railway with comprehensive error handling and debugging capabilities.

## Features

- PDF to JSON conversion using Docling
- OCR support for scanned documents
- Table extraction
- Comprehensive error handling with detailed error codes
- Request tracking with unique IDs
- Debug mode for troubleshooting
- Health check endpoint
- Docker-ready for Railway deployment

## API Endpoints

### `GET /`
Root endpoint with API information.

### `GET /health`
Health check endpoint returning system status and Docling availability.

**Response Example:**
```json
{
  "status": "healthy",
  "docling_available": true,
  "version": "1.0.0",
  "timestamp": "2024-01-15T10:30:00Z",
  "environment": {
    "python_version": "3.11.0",
    "docling_status": "available",
    "converter_status": "initialized",
    "debug_mode": "false"
  }
}
```

### `POST /convert`
Convert a PDF file to structured JSON.

**Request:**
- Method: `POST`
- Content-Type: `multipart/form-data`
- Body: `file` - PDF file to convert

**cURL Example:**
```bash
curl -X POST "https://your-app.railway.app/convert" \
  -H "accept: application/json" \
  -F "file=@document.pdf"
```

**Success Response:**
```json
{
  "success": true,
  "request_id": "req_abc123def456",
  "filename": "document.pdf",
  "conversion_time_seconds": 2.345,
  "document": {
    "pages": [...],
    "texts": [...],
    "tables": [...],
    ...
  },
  "metadata": {
    "file_size_bytes": 102400,
    "conversion_timestamp": "2024-01-15T10:30:00Z",
    "markdown_preview": "...",
    "has_tables": true,
    "has_pictures": false,
    "page_count": 5
  }
}
```

### `POST /convert/markdown`
Convert a PDF file to Markdown format.

### `GET /debug/info`
Debug endpoint (only available when `DEBUG=true`).

## Error Handling

All errors return a structured response with detailed information:

```json
{
  "success": false,
  "error": {
    "error_code": "CONVERSION_ERROR",
    "message": "Failed to convert PDF",
    "details": "Detailed error information",
    "timestamp": "2024-01-15T10:30:00Z",
    "request_id": "req_abc123def456",
    "traceback": "..." // Only when DEBUG=true
  }
}
```

### Error Codes

| Code | HTTP Status | Description |
|------|-------------|-------------|
| `DOCLING_UNAVAILABLE` | 503 | Docling library not installed/available |
| `CONVERTER_NOT_INITIALIZED` | 503 | Document converter failed to initialize |
| `INVALID_FILE_TYPE` | 400 | Uploaded file is not a PDF |
| `FILE_READ_ERROR` | 400 | Failed to read uploaded file |
| `CONVERSION_ERROR` | 500 | PDF conversion failed |
| `SERIALIZATION_ERROR` | 500 | Failed to serialize result to JSON |
| `INTERNAL_SERVER_ERROR` | 500 | Unexpected server error |

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | `8000` | Server port (Railway sets this automatically) |
| `HOST` | `0.0.0.0` | Server host |
| `DEBUG` | `false` | Enable debug mode for verbose logging |

## Railway Deployment

### Quick Deploy

1. Fork or clone this repository
2. Connect your Railway account to GitHub
3. Create a new project from this repository
4. Railway will automatically detect the Dockerfile and deploy

### Environment Setup in Railway

1. Go to your Railway project settings
2. Add environment variables:
   - `DEBUG=false` (or `true` for debugging)

### Monitoring

- Check `/health` endpoint for service status
- View logs in Railway dashboard
- Enable `DEBUG=true` for detailed logging

## Local Development

### Prerequisites

- Python 3.11+
- pip

### Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Run the server
python -m uvicorn app.main:app --reload --port 8000
```

### Docker Local Testing

```bash
# Build the image
docker build -t pdf-converter .

# Run the container
docker run -p 8000:8000 -e DEBUG=true pdf-converter
```

## Debugging Tips

1. **Enable Debug Mode**: Set `DEBUG=true` to get:
   - Verbose logging with request/response details
   - Stack traces in error responses
   - Access to `/debug/info` endpoint

2. **Check Health Endpoint**: The `/health` endpoint shows:
   - Docling availability status
   - Converter initialization status
   - Python version and environment info

3. **Request Tracking**: Every request gets a unique `request_id`:
   - Included in response headers as `X-Request-ID`
   - Present in all log messages
   - Included in error responses

4. **Common Issues**:
   - `DOCLING_UNAVAILABLE`: Check if all system dependencies are installed
   - `CONVERTER_NOT_INITIALIZED`: Check startup logs for initialization errors
   - `CONVERSION_ERROR`: Check if the PDF is valid and not corrupted

## License

MIT
