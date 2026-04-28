"""
Resume Insight Pro — Flask backend.

Run locally:
    pip install -r requirements.txt
    export GEMINI_API_KEY=your_key_here
    export SESSION_SECRET=any_random_string
    python app.py

Deploy on Render:
    Build command:  pip install -r requirements.txt
    Start command:  gunicorn app:app
    Environment:    GEMINI_API_KEY, SESSION_SECRET (required)

Optionally serves the built React frontend from
artifacts/resume-analyzer/dist/ if it exists at the same root.
"""

from __future__ import annotations

import json
import os
import re
import uuid
from datetime import datetime, timezone
from io import BytesIO
from typing import Any

from flask import Flask, jsonify, request, session, send_from_directory
from flask_cors import CORS
from google import genai
from google.genai import types as genai_types
from pypdf import PdfReader


# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

FRONTEND_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "artifacts",
    "resume-analyzer",
    "dist",
)

app = Flask(
    __name__,
    static_folder=FRONTEND_DIR if os.path.isdir(FRONTEND_DIR) else None,
    static_url_path="",
)

app.secret_key = os.environ.get("SESSION_SECRET")
if not app.secret_key:
    raise RuntimeError("SESSION_SECRET environment variable is required.")

app.config.update(
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
    MAX_CONTENT_LENGTH=10 * 1024 * 1024,  # 10 MB upload cap
)

CORS(app, supports_credentials=True)

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY environment variable is required.")

client = genai.Client(api_key=GEMINI_API_KEY)


# ---------------------------------------------------------------------------
# Session-scoped in-memory store
#   Each browser session gets its own private list of analyzed resumes.
#   This mirrors the original Node implementation (express-session + MemoryStore).
# ---------------------------------------------------------------------------

SESSIONS: dict[str, list[dict[str, Any]]] = {}


def _sid() -> str:
    if "sid" not in session:
        session["sid"] = str(uuid.uuid4())
        session.permanent = True
    return session["sid"]


def _resumes() -> list[dict[str, Any]]:
    return SESSIONS.setdefault(_sid(), [])


# ---------------------------------------------------------------------------
# PDF text extraction
# ---------------------------------------------------------------------------

def extract_pdf_text(buffer: bytes) -> str:
    reader = PdfReader(BytesIO(buffer))
    chunks: list[str] = []
    for page in reader.pages:
        try:
            chunks.append(page.extract_text() or "")
        except Exception:
            continue
    return "\n".join(chunks).strip()


def count_words(text: str) -> int:
    return len([w for w in re.split(r"\s+", text) if w])


# ---------------------------------------------------------------------------
# Gemini analysis
# ---------------------------------------------------------------------------

RESPONSE_SCHEMA = genai_types.Schema(
    type=genai_types.Type.OBJECT,
    required=[
        "overallScore",
        "atsScore",
        "skillsMatchScore",
        "experienceScore",
        "experienceLevel",
        "yearsOfExperience",
        "summary",
        "strengths",
        "weaknesses",
        "matchedSkills",
        "missingSkills",
        "skillCategories",
        "keywordHits",
        "improvements",
        "atsBreakdown",
    ],
    properties={
        "overallScore": genai_types.Schema(type=genai_types.Type.INTEGER),
        "atsScore": genai_types.Schema(type=genai_types.Type.INTEGER),
        "skillsMatchScore": genai_types.Schema(type=genai_types.Type.INTEGER),
        "experienceScore": genai_types.Schema(type=genai_types.Type.INTEGER),
        "experienceLevel": genai_types.Schema(
            type=genai_types.Type.STRING,
            enum=["entry", "mid", "senior", "expert"],
        ),
        "yearsOfExperience": genai_types.Schema(type=genai_types.Type.NUMBER),
        "summary": genai_types.Schema(type=genai_types.Type.STRING),
        "strengths": genai_types.Schema(
            type=genai_types.Type.ARRAY,
            items=genai_types.Schema(type=genai_types.Type.STRING),
        ),
        "weaknesses": genai_types.Schema(
            type=genai_types.Type.ARRAY,
            items=genai_types.Schema(type=genai_types.Type.STRING),
        ),
        "matchedSkills": genai_types.Schema(
            type=genai_types.Type.ARRAY,
            items=genai_types.Schema(type=genai_types.Type.STRING),
        ),
        "missingSkills": genai_types.Schema(
            type=genai_types.Type.ARRAY,
            items=genai_types.Schema(type=genai_types.Type.STRING),
        ),
        "skillCategories": genai_types.Schema(
            type=genai_types.Type.ARRAY,
            items=genai_types.Schema(
                type=genai_types.Type.OBJECT,
                required=["name", "count", "skills"],
                properties={
                    "name": genai_types.Schema(type=genai_types.Type.STRING),
                    "count": genai_types.Schema(type=genai_types.Type.INTEGER),
                    "skills": genai_types.Schema(
                        type=genai_types.Type.ARRAY,
                        items=genai_types.Schema(type=genai_types.Type.STRING),
                    ),
                },
            ),
        ),
        "keywordHits": genai_types.Schema(
            type=genai_types.Type.ARRAY,
            items=genai_types.Schema(
                type=genai_types.Type.OBJECT,
                required=["keyword", "matched", "occurrences"],
                properties={
                    "keyword": genai_types.Schema(type=genai_types.Type.STRING),
                    "matched": genai_types.Schema(type=genai_types.Type.BOOLEAN),
                    "occurrences": genai_types.Schema(type=genai_types.Type.INTEGER),
                },
            ),
        ),
        "improvements": genai_types.Schema(
            type=genai_types.Type.ARRAY,
            items=genai_types.Schema(
                type=genai_types.Type.OBJECT,
                required=[
                    "title",
                    "category",
                    "priority",
                    "current",
                    "suggested",
                    "rationale",
                ],
                properties={
                    "title": genai_types.Schema(type=genai_types.Type.STRING),
                    "category": genai_types.Schema(type=genai_types.Type.STRING),
                    "priority": genai_types.Schema(
                        type=genai_types.Type.STRING,
                        enum=["high", "medium", "low"],
                    ),
                    "current": genai_types.Schema(type=genai_types.Type.STRING),
                    "suggested": genai_types.Schema(type=genai_types.Type.STRING),
                    "rationale": genai_types.Schema(type=genai_types.Type.STRING),
                },
            ),
        ),
        "atsBreakdown": genai_types.Schema(
            type=genai_types.Type.OBJECT,
            required=[
                "formatting",
                "keywords",
                "readability",
                "sectionCoverage",
                "contactInfo",
            ],
            properties={
                "formatting": genai_types.Schema(type=genai_types.Type.INTEGER),
                "keywords": genai_types.Schema(type=genai_types.Type.INTEGER),
                "readability": genai_types.Schema(type=genai_types.Type.INTEGER),
                "sectionCoverage": genai_types.Schema(type=genai_types.Type.INTEGER),
                "contactInfo": genai_types.Schema(type=genai_types.Type.INTEGER),
            },
        ),
    },
)


SYSTEM_INSTRUCTION = """You are an expert technical recruiter and ATS (Applicant Tracking System) specialist who scores resumes with rigor and provides specific, actionable feedback.

Scoring rubric (be honest and discerning — do NOT give every resume an 80+):
- overallScore: holistic 0-100 quality of the resume considering content, structure, achievements, clarity. Most real resumes score 50-75; only exceptional resumes exceed 85.
- atsScore: 0-100 representing how well an ATS would parse and match this resume against the job description. If no job description is provided, score against general ATS best practices (clear sections, standard fonts implied, no tables/columns indicators, contact info, dates, role titles).
- skillsMatchScore: 0-100. If a job description exists, this is the percentage of required skills the resume credibly demonstrates. If no JD, this represents skill breadth and depth signals from the resume.
- experienceScore: 0-100 representing the strength and relevance of work experience.
- experienceLevel: one of "entry" (0-2 yrs), "mid" (3-5 yrs), "senior" (6-10 yrs), "expert" (10+ yrs).
- yearsOfExperience: numeric estimate (use decimals if needed, e.g., 4.5).

Content rules:
- summary: 2-3 sentence professional summary of the candidate written for a hiring manager.
- strengths: 3-5 specific strengths grounded in the resume content (not generic).
- weaknesses: 3-5 honest gaps or weaknesses; if a JD exists, focus on misalignment with it.
- matchedSkills: skills present in the resume that match the job description (or are clearly valuable if no JD). Aim for 8-15.
- missingSkills: skills required/preferred by the JD that are absent (or commonly expected for this role type if no JD). Aim for 4-10.
- skillCategories: group all detected skills into 4-6 categories (e.g., "Languages", "Frameworks", "Cloud & DevOps", "Data", "Soft Skills"). Each category must have at least 1 skill.
- keywordHits: 8-15 important keywords from the JD (or expected for the role) and whether they appear in the resume, with the number of occurrences. If no JD, infer keywords from common expectations.
- improvements: 4-6 concrete, prioritized improvement suggestions. Each must include the actual current text from the resume (or a faithful paraphrase if exact text isn't available) and a stronger suggested rewrite. Mix priorities.
- atsBreakdown: 0-100 sub-scores for each ATS dimension. Be granular; these should not all equal atsScore.

Return ONLY the JSON object specified by the schema. No prose, no code fences."""


def _clamp(value: Any) -> int:
    try:
        v = int(round(float(value)))
    except (TypeError, ValueError):
        return 0
    return max(0, min(100, v))


def analyze_resume(
    resume_text: str,
    job_description: str | None,
    candidate_name: str | None,
    file_name: str,
) -> dict[str, Any]:
    user_prompt_parts = [
        f"Analyze the following resume{' against the provided job description' if job_description else ''}.",
        "",
    ]
    if candidate_name:
        user_prompt_parts.append(f"Candidate label: {candidate_name}")
    user_prompt_parts.append(f"File: {file_name}")
    user_prompt_parts.extend(
        [
            "",
            "=== RESUME TEXT ===",
            resume_text[:18000],
            "=== END RESUME ===",
            "",
        ]
    )
    if job_description:
        user_prompt_parts.extend(
            [
                "=== JOB DESCRIPTION ===",
                job_description[:6000],
                "=== END JOB DESCRIPTION ===",
            ]
        )
    else:
        user_prompt_parts.append(
            "No job description was provided. Score the resume against general best practices and infer the most likely target role from its content."
        )
    user_prompt_parts.extend(["", "Return the structured JSON analysis."])

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents="\n".join(user_prompt_parts),
        config=genai_types.GenerateContentConfig(
            system_instruction=SYSTEM_INSTRUCTION,
            response_mime_type="application/json",
            response_schema=RESPONSE_SCHEMA,
            max_output_tokens=8192,
            temperature=0.4,
        ),
    )

    raw = response.text
    if not raw:
        raise RuntimeError("Gemini returned an empty response")

    import json

    parsed: dict[str, Any] = json.loads(raw)

    breakdown = parsed.get("atsBreakdown") or {}
    return {
        "overallScore": _clamp(parsed.get("overallScore")),
        "atsScore": _clamp(parsed.get("atsScore")),
        "skillsMatchScore": _clamp(parsed.get("skillsMatchScore")),
        "experienceScore": _clamp(parsed.get("experienceScore")),
        "experienceLevel": parsed.get("experienceLevel") or "entry",
        "yearsOfExperience": max(0.0, float(parsed.get("yearsOfExperience") or 0)),
        "summary": parsed.get("summary") or "",
        "strengths": parsed.get("strengths") or [],
        "weaknesses": parsed.get("weaknesses") or [],
        "matchedSkills": parsed.get("matchedSkills") or [],
        "missingSkills": parsed.get("missingSkills") or [],
        "skillCategories": parsed.get("skillCategories") or [],
        "keywordHits": parsed.get("keywordHits") or [],
        "improvements": parsed.get("improvements") or [],
        "atsBreakdown": {
            "formatting": _clamp(breakdown.get("formatting")),
            "keywords": _clamp(breakdown.get("keywords")),
            "readability": _clamp(breakdown.get("readability")),
            "sectionCoverage": _clamp(breakdown.get("sectionCoverage")),
            "contactInfo": _clamp(breakdown.get("contactInfo")),
        },
    }


# ---------------------------------------------------------------------------
# API routes
# ---------------------------------------------------------------------------

@app.get("/api/healthz")
def healthz():
    return jsonify({"status": "ok"})


@app.get("/api/resumes")
def list_resumes():
    return jsonify(_resumes())


@app.delete("/api/resumes")
def clear_resumes():
    SESSIONS[_sid()] = []
    return ("", 204)


@app.get("/api/resumes/summary")
def resumes_summary():
    resumes = _resumes()
    total = len(resumes)
    levels = ["entry", "mid", "senior", "expert"]

    if total == 0:
        return jsonify(
            {
                "totalResumes": 0,
                "averageOverall": 0,
                "averageAts": 0,
                "averageSkillsMatch": 0,
                "topCandidate": None,
                "levelBreakdown": [{"level": lvl, "count": 0} for lvl in levels],
            }
        )

    def avg(key: str) -> int:
        return round(sum(r[key] for r in resumes) / total)

    top = max(resumes, key=lambda r: r["overallScore"])
    return jsonify(
        {
            "totalResumes": total,
            "averageOverall": avg("overallScore"),
            "averageAts": avg("atsScore"),
            "averageSkillsMatch": avg("skillsMatchScore"),
            "topCandidate": top.get("candidateName") or top.get("fileName"),
            "levelBreakdown": [
                {
                    "level": lvl,
                    "count": sum(1 for r in resumes if r.get("experienceLevel") == lvl),
                }
                for lvl in levels
            ],
        }
    )


@app.post("/api/resumes/analyze")
def analyze_endpoint():
    file = request.files.get("file")
    if file is None or file.filename == "":
        return jsonify({"error": "A PDF file is required (field name: file)."}), 400

    if not (
        (file.mimetype or "").lower().endswith("pdf")
        or file.filename.lower().endswith(".pdf")
    ):
        return jsonify({"error": "Only PDF files are supported."}), 400

    job_description = (request.form.get("jobDescription") or "").strip() or None
    candidate_name = (request.form.get("candidateName") or "").strip() or None

    buffer = file.read()
    try:
        resume_text = extract_pdf_text(buffer)
    except Exception as err:  # noqa: BLE001
        app.logger.exception("PDF extraction failed")
        return (
            jsonify(
                {
                    "error": "Could not extract text from this PDF. It may be scanned, encrypted, or corrupted.",
                    "detail": str(err),
                }
            ),
            400,
        )

    if not resume_text or len(resume_text) < 50:
        return (
            jsonify(
                {
                    "error": "The PDF appears to contain no extractable text. If it is a scanned document, please upload a text-based PDF."
                }
            ),
            400,
        )

    try:
        analysis = analyze_resume(
            resume_text=resume_text,
            job_description=job_description,
            candidate_name=candidate_name,
            file_name=file.filename,
        )
    except Exception as err:  # noqa: BLE001
        app.logger.exception("Gemini analysis failed")
        return jsonify({"error": f"Analysis failed: {err}"}), 502

    stored = {
        "id": str(uuid.uuid4()),
        "fileName": file.filename,
        "candidateName": candidate_name,
        "createdAt": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "jobDescription": job_description,
        "textPreview": resume_text[:1200],
        "wordCount": count_words(resume_text),
        **analysis,
    }
    _resumes().append(stored)
    return jsonify(stored)


@app.get("/api/resumes/<resume_id>")
def get_resume(resume_id: str):
    for r in _resumes():
        if r["id"] == resume_id:
            return jsonify(r)
    return jsonify({"error": "Resume not found in this session."}), 404


@app.delete("/api/resumes/<resume_id>")
def delete_resume(resume_id: str):
    items = _resumes()
    for i, r in enumerate(items):
        if r["id"] == resume_id:
            items.pop(i)
            return ("", 204)
    return jsonify({"error": "Resume not found in this session."}), 404


@app.post("/api/resumes/compare")
def compare_resumes():
    payload = request.get_json(silent=True) or {}
    ids = payload.get("ids") or []
    if not isinstance(ids, list):
        return jsonify({"error": "Expected `ids` to be an array of resume ids."}), 400

    by_id = {r["id"]: r for r in _resumes()}
    selected = [by_id[i] for i in ids if i in by_id]

    if len(selected) < 2:
        return (
            jsonify(
                {"error": "At least two valid resumes are required for comparison."}
            ),
            400,
        )

    rows = [
        {
            "id": r["id"],
            "label": r.get("candidateName") or r.get("fileName"),
            "overallScore": r["overallScore"],
            "atsScore": r["atsScore"],
            "skillsMatchScore": r["skillsMatchScore"],
            "experienceScore": r["experienceScore"],
            "experienceLevel": r["experienceLevel"],
            "matchedSkills": r["matchedSkills"],
            "missingSkills": r["missingSkills"],
        }
        for r in selected
    ]

    skill_sets = [{s.lower() for s in r["matchedSkills"]} for r in selected]

    seen: set[str] = set()
    shared_skills: list[str] = []
    for r in selected:
        for skill in r["matchedSkills"]:
            lower = skill.lower()
            if lower in seen:
                continue
            if all(lower in s for s in skill_sets):
                shared_skills.append(skill)
                seen.add(lower)

    unique_skills = []
    for i, r in enumerate(selected):
        others = [s for j, s in enumerate(skill_sets) if j != i]
        unique = [
            skill
            for skill in r["matchedSkills"]
            if not any(skill.lower() in s for s in others)
        ]
        unique_skills.append({"id": r["id"], "skills": unique})

    winner = max(selected, key=lambda r: r["overallScore"])
    return jsonify(
        {
            "rows": rows,
            "sharedSkills": shared_skills,
            "uniqueSkills": unique_skills,
            "winnerId": winner["id"],
        }
    )


# ---------------------------------------------------------------------------
# Frontend (optional) — serves the built React app if present.
# ---------------------------------------------------------------------------

@app.route("/", defaults={"path": ""})
@app.route("/<path:path>")
def serve_frontend(path: str):
    if not os.path.isdir(FRONTEND_DIR):
        return (
            jsonify(
                {
                    "error": "Frontend not built.",
                    "hint": "Build the frontend (e.g. `pnpm --filter @workspace/resume-analyzer run build`) so that artifacts/resume-analyzer/dist/ exists, or deploy the API separately.",
                }
            ),
            404,
        )

    if path and os.path.isfile(os.path.join(FRONTEND_DIR, path)):
        return send_from_directory(FRONTEND_DIR, path)
    return send_from_directory(FRONTEND_DIR, "index.html")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=False)
