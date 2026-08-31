"""
PDF Quiz Generator
-------------------
A Streamlit app that lets a student upload a PDF (course notes, textbook
snippet, etc.), extracts the text, and uses Google's Gemini API to generate
a configurable multiple-choice quiz. The quiz is displayed with radio
buttons and can be graded on submit.

Run with:
    streamlit run app.py

Dependencies:
    pip install streamlit google-genai pypdf
"""

import json

import streamlit as st
from pypdf import PdfReader
from google import genai
from google.genai import types


# --------------------------------------------------------------------------
# Page config
# --------------------------------------------------------------------------
st.set_page_config(page_title="PDF Quiz Generator", page_icon="📝", layout="centered")
st.title("📝 PDF Quiz Generator")
st.caption("Upload a PDF of your notes and get a custom multiple-choice quiz, powered by Gemini.")


# --------------------------------------------------------------------------
# Session state initialization
# --------------------------------------------------------------------------
if "quiz" not in st.session_state:
    st.session_state.quiz = None          # list of question dicts once generated
if "submitted" not in st.session_state:
    st.session_state.submitted = False    # whether the quiz has been graded
if "pdf_text" not in st.session_state:
    st.session_state.pdf_text = ""


QUESTION_TYPE_OPTIONS = [
    "Conceptual / understanding",
    "Definition / terminology",
    "Application / scenario-based",
    "Comparison / contrast",
    "Calculation / numeric",
    "Cause and effect",
    "True-or-false style (as MCQ)",
]

DIFFICULTY_OPTIONS = ["Easy", "Medium", "Hard", "Mixed (easy to hard)"]


# --------------------------------------------------------------------------
# Sidebar: API key + core settings
# --------------------------------------------------------------------------
with st.sidebar:
    st.header("Settings")
    api_key = st.text_input(
        "Gemini API Key",
        type="password",
        help="Get a key from https://aistudio.google.com/apikey. It is only used for this session and is never stored.",
    )
    model_name = st.selectbox(
        "Model",
        ["gemini-3.6-flash", "gemini-3.7-flash", "gemini-2.5-flash"],
        index=0,
        help="Gemini 3.6 Flash is the current default; 3.7 Flash is newer/more capable, 2.5 Flash is the older fallback.",
    )
    st.divider()
    st.caption("Your API key is kept in memory for this session only and is not saved anywhere.")


# --------------------------------------------------------------------------
# Quiz configuration (main page, above the uploader)
# --------------------------------------------------------------------------
st.subheader("Quiz Settings")

num_questions = st.slider("Number of questions", min_value=1, max_value=25, value=5)

question_types = st.multiselect(
    "Question types to include",
    QUESTION_TYPE_OPTIONS,
    default=["Conceptual / understanding", "Definition / terminology"],
    help="Gemini will try to mix questions across the types you pick. Leave empty to let it choose freely.",
)

difficulty = st.select_slider(
    "Difficulty",
    options=DIFFICULTY_OPTIONS,
    value="Medium",
)

with st.expander("Advanced controls"):
    num_options = st.slider(
        "Answer options per question",
        min_value=2,
        max_value=6,
        value=4,
        help="E.g. 2 for true/false-style questions, 4 for classic MCQ.",
    )
    focus_topic = st.text_input(
        "Focus on a specific topic/section (optional)",
        placeholder="e.g. Only chapter 3, or 'photosynthesis'",
        help="Leave blank to draw questions from the whole document.",
    )
    avoid_topics = st.text_input(
        "Topics to avoid (optional)",
        placeholder="e.g. dates, footnotes, appendix material",
    )
    language = st.text_input(
        "Quiz language",
        value="English",
        help="The language the questions, options, and explanations should be written in.",
    )
    temperature = st.slider(
        "Creativity (temperature)",
        min_value=0.0,
        max_value=1.5,
        value=0.7,
        step=0.1,
        help="Lower = more literal/predictable questions. Higher = more varied phrasing, at some risk of drift from the source text.",
    )
    max_chars = st.number_input(
        "Max characters of PDF text sent to the model",
        min_value=1000,
        max_value=200000,
        value=30000,
        step=1000,
        help="Very long PDFs are truncated to this many characters to control cost/latency.",
    )
    custom_instructions = st.text_area(
        "Extra instructions for the quiz generator (optional)",
        placeholder="e.g. 'Write distractors that reflect common student misconceptions' or 'Keep questions under 20 words'",
    )


# --------------------------------------------------------------------------
# Helper: extract text from an uploaded PDF
# --------------------------------------------------------------------------
def extract_pdf_text(uploaded_file) -> str:
    reader = PdfReader(uploaded_file)
    pages_text = []
    for page in reader.pages:
        text = page.extract_text() or ""
        pages_text.append(text)
    return "\n".join(pages_text).strip()


# --------------------------------------------------------------------------
# Helper: build the JSON schema dynamically based on user settings
# --------------------------------------------------------------------------
def build_quiz_schema(n_questions: int, n_options: int) -> dict:
    return {
        "type": "object",
        "properties": {
            "questions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "question": {"type": "string"},
                        "question_type": {"type": "string"},
                        "options": {
                            "type": "array",
                            "items": {"type": "string"},
                            "minItems": n_options,
                            "maxItems": n_options,
                        },
                        "correct_index": {"type": "integer"},
                        "explanation": {"type": "string"},
                    },
                    "required": ["question", "options", "correct_index", "explanation"],
                },
                "minItems": n_questions,
                "maxItems": n_questions,
            }
        },
        "required": ["questions"],
    }


# --------------------------------------------------------------------------
# Helper: call Gemini to generate a quiz as structured JSON
# --------------------------------------------------------------------------
def generate_quiz(
    client: genai.Client,
    model_name: str,
    source_text: str,
    n_questions: int,
    n_options: int,
    q_types: list,
    difficulty: str,
    focus_topic: str,
    avoid_topics: str,
    language: str,
    temperature: float,
    max_chars: int,
    custom_instructions: str,
) -> list:
    trimmed = source_text[:max_chars]

    type_line = (
        f"- Draw questions from these question types, mixing them across the quiz: {', '.join(q_types)}."
        if q_types
        else "- Choose a sensible mix of question types on your own."
    )
    focus_line = f"- Focus primarily on this topic/section: {focus_topic}." if focus_topic.strip() else ""
    avoid_line = f"- Avoid asking about: {avoid_topics}." if avoid_topics.strip() else ""
    custom_line = f"- Additional instructions: {custom_instructions}" if custom_instructions.strip() else ""

    prompt = f"""You are a helpful teaching assistant. Based ONLY on the study
material below, write exactly {n_questions} multiple-choice questions to help
a student test their understanding.

Rules:
- Each question must have exactly {n_options} answer options.
- Exactly one option must be correct.
- correct_index is the 0-based index of the correct option in the options list.
- Difficulty level should be: {difficulty}.
- Include a brief explanation (1-2 sentences) for why the correct answer is right.
- Set "question_type" on each question to whichever type it best matches.
- Write the question, options, and explanation in: {language}.
- Cover a range of concepts from the material, not just the first section.
- Do not ask about material that isn't present in the text below.
{type_line}
{focus_line}
{avoid_line}
{custom_line}

STUDY MATERIAL:
\"\"\"
{trimmed}
\"\"\"
"""

    response = client.models.generate_content(
        model=model_name,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=build_quiz_schema(n_questions, n_options),
            temperature=temperature,
        ),
    )

    data = json.loads(response.text)
    return data["questions"]


# --------------------------------------------------------------------------
# Main flow: upload -> extract -> generate
# --------------------------------------------------------------------------
st.divider()
uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

col1, col2 = st.columns([1, 1])
with col1:
    generate_clicked = st.button("Generate Quiz", type="primary", use_container_width=True)
with col2:
    reset_clicked = st.button("Reset", use_container_width=True)

if reset_clicked:
    st.session_state.quiz = None
    st.session_state.submitted = False
    st.session_state.pdf_text = ""
    st.rerun()

if generate_clicked:
    if not api_key:
        st.error("Please enter your Gemini API key in the sidebar.")
    elif not uploaded_file:
        st.error("Please upload a PDF first.")
    else:
        with st.spinner("Extracting text from PDF..."):
            try:
                pdf_text = extract_pdf_text(uploaded_file)
            except Exception as e:
                st.error(f"Could not read the PDF: {e}")
                pdf_text = ""

        if pdf_text and len(pdf_text.strip()) < 50:
            st.warning(
                "Very little text was found in this PDF. It may be a scanned "
                "image without a text layer, so the quiz quality may suffer."
            )

        if pdf_text:
            st.session_state.pdf_text = pdf_text
            try:
                with st.spinner(f"Generating {num_questions}-question quiz with Gemini..."):
                    client = genai.Client(api_key=api_key)
                    questions = generate_quiz(
                        client,
                        model_name,
                        pdf_text,
                        n_questions=num_questions,
                        n_options=num_options,
                        q_types=question_types,
                        difficulty=difficulty,
                        focus_topic=focus_topic,
                        avoid_topics=avoid_topics,
                        language=language,
                        temperature=temperature,
                        max_chars=int(max_chars),
                        custom_instructions=custom_instructions,
                    )
                st.session_state.quiz = questions
                st.session_state.submitted = False
                st.success(f"Quiz generated from {len(pdf_text)} characters of extracted text!")
            except Exception as e:
                st.error(f"Failed to generate quiz: {e}")


# --------------------------------------------------------------------------
# Display quiz
# --------------------------------------------------------------------------
if st.session_state.quiz:
    st.divider()
    st.subheader("Your Quiz")

    with st.form("quiz_form"):
        user_answers = {}
        for i, q in enumerate(st.session_state.quiz):
            type_tag = f"  \n*Type: {q['question_type']}*" if q.get("question_type") else ""
            st.markdown(f"**Q{i + 1}. {q['question']}**{type_tag}")
            user_answers[i] = st.radio(
                label=f"Options for question {i + 1}",
                options=list(range(len(q["options"]))),
                format_func=lambda idx, opts=q["options"]: opts[idx],
                key=f"q_{i}",
                index=None,
                label_visibility="collapsed",
            )
            st.write("")

        submit = st.form_submit_button("Submit Answers", type="primary")

    if submit:
        st.session_state.submitted = True
        st.session_state.user_answers = user_answers

    if st.session_state.submitted:
        st.divider()
        st.subheader("Results")

        score = 0
        total = len(st.session_state.quiz)
        answers = st.session_state.get("user_answers", {})

        for i, q in enumerate(st.session_state.quiz):
            chosen = answers.get(i)
            correct = q["correct_index"]
            is_correct = chosen == correct

            if is_correct:
                score += 1

            st.markdown(f"**Q{i + 1}. {q['question']}**")
            if chosen is None:
                st.warning("No answer selected.")
            elif is_correct:
                st.success(f"✅ Correct: {q['options'][chosen]}")
            else:
                st.error(f"❌ Your answer: {q['options'][chosen]}")
                st.info(f"Correct answer: {q['options'][correct]}")

            st.caption(f"💡 {q['explanation']}")
            st.write("")

        st.metric("Score", f"{score} / {total}")
        if score == total:
            st.balloons()
else:
    st.info("Configure your quiz settings, upload a PDF, and click **Generate Quiz** to get started.")
