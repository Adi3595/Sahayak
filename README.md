# Sahayak (Groq) Backend

This project is a FastAPI backend that powers an **AI-assisted education web app**. It generates:

- Lesson plans
- Differentiated worksheets (PDF/DOCX)
- PDF summaries (with page range)
- Fun learning activities
- Blackboard-style diagram instructions + generated images
- Knowledge-base style concept explanations
- Student progress tracking and analytics
- Downloadable reports (PDF/DOCX)

## Tech Stack

- **FastAPI** (API server)
- **Groq API** (LLM: `llama-3.3-70b-versatile`)
- **HTML/PDF/DOCX generation**
  - PDF: `fpdf` (with Unicode handling; includes fallback logic)
  - DOCX: `python-docx`
- **OCR/text extraction** (for uploaded files)
  - `PyPDF2`, `python-docx`, `pytesseract` (images)
- **Visual generation**
  - `matplotlib` (diagram images encoded as base64)

## Project Structure (high level)

- `main.py` — FastAPI app + all API endpoints
- `utils.py` — upload, text extraction, worksheet/document creation helpers
- `public/` — frontend HTML pages served by the backend routes
- `templates/` — reserved for future/template usage
- `uploads/` — user uploads (cleaned up automatically)
- `outputs/` — generated PDFs/DOCXs/archives (downloadable)

## Prerequisites

- Python 3.9+
- A Groq API key
- (Optional, for OCR) Tesseract installed on the system

## Setup

### 1) Create a virtual environment

```bash
python -m venv .venv
```

### 2) Activate it

Windows (cmd):

```bash
.venv\Scripts\activate
```

### 3) Install dependencies

```bash
pip install -r requirements.txt
```

### 4) Configure environment variables

Create a `.env` file (or set environment variable) with:

- `GROQ_API_KEY`

Example:

```bash
GROQ_API_KEY=your_key_here
```

## Run the server

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The app also contains a basic `if __name__ == "__main__": app.run()` fallback, but `uvicorn` is recommended.

## API Overview

### Health
- `GET /health`

### Frontend page routes (serve HTML files)
- `GET /` (landing)
- `GET /lesson-plan`
- `GET /differentiated-materials`
- `GET /fun-activities`
- `GET /blackboard-diagrams`
- `GET /student-progress`
- `GET /knowledge-base`

### Lesson plans & chat
- `POST /generate/`
  - Detects lesson-plan requests automatically using keywords.
  - Uses `conversation_id` to keep short in-memory conversation history.

### PDF summarization
- `POST /summarize-pdf/`
  - Form fields:
    - `file` (PDF upload)
    - `summary_length` = `short|medium|detailed` (default `medium`)
    - `page_start` (default `1`)
    - `page_end` (optional)
  - Returns JSON with `summary` and `download_url`.

### Worksheet generation (from uploaded file)
- `POST /generate-worksheets/`
  - Form fields:
    - `file` (PDF/DOCX/etc. upload)
    - `subject` (optional)
    - `grades` (comma-separated, e.g. `Grade 5,Grade 6`)
    - `difficulty` (default `moderate`)
    - `question_types` (comma-separated; default `mcq,theory`)
    - `page_start`, `page_end` (optional)
    - `output_format` (currently used for response; PDFs/DOCXs are generated)

### Worksheet generation (from topics)
- `POST /generate-from-topics/`
  - Form fields:
    - `main_topic`
    - `subject`
    - `grades`
    - `difficulty`
    - `question_types`
    - `subtopics` (optional, comma-separated)

### Fun activities
- `POST /generate-activity/`
  - Form fields:
    - `subject`, `topic`, `grade_level`
    - optional: `activity_type`, `duration`, `resources`, `learning_objective`

### Blackboard diagrams
- `POST /generate-visual-aids/`
  - Produces diagram instructions + generated images (base64 PNGs).
- `POST /generate-diagram/`
  - Simpler diagram generator with LLM instructions + generated sample images.
- `POST /download-diagrams/`
  - Accepts base64 diagram images and returns a ZIP download URL.

### Knowledge base
- `POST /explain-concept/`
  - Form/body (JSON):
    - `concept`, `grade_level`, `language`, `include_analogies`, `include_examples`, `include_visuals`
  - Also generates `related_concepts`.

### Student progress tracking
In-memory data store:
- `GET /api/students/` (filters supported: class/status/grade)
- `POST /api/students/` (create)
- `GET /api/students/{student_id}` (details + computed statistics)
- `POST /api/progress/` (update assignment/test results)
- `GET /api/progress/overview` (class overview + grade distribution + trends)
- `POST /api/progress/bulk-update` (bulk progress update)
- `POST /api/reports/generate` (PDF/DOCX report generation)

### Downloads
- `GET /download/{filename}`

### Conversations (in-memory)
- `GET /conversation/{conversation_id}`
- `GET /conversations/`
- `DELETE /conversation/{conversation_id}`

## Notes / Behavior

- **In-memory storage only**: student and conversation data is stored in process memory (`student_records`, `progress_history`, `conversations`). Restarting the server resets data.
- **Uploads cleanup**: background scheduler deletes old files from `uploads/` and `outputs/` (older than ~24 hours).
- **Diagram generation**: images are generated via matplotlib and returned as base64 data URIs.

## Deploying (Firebase Hosting)

The repository includes Firebase configuration files such as `firebase.json` and `.firebaserc`. Adjustments may be required depending on whether you deploy the backend separately (e.g., Cloud Run/Functions) or via a single hosting configuration.

## Contributors :
Aditya Gawali

