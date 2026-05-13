import streamlit as st
import json
import time
import random
from datetime import datetime
import requests


st.set_page_config(
    page_title="EduAI Tutor",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)


with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_API_KEY = "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"

MODEL = "llama-3.3-70b-versatile"

SUBJECTS = {
    "Mathematics": ["Algebra", "Calculus", "Statistics", "Geometry", "Linear Algebra"],
    "Science": ["Physics", "Chemistry", "Biology", "Astronomy", "Environmental Science"],
    "Computer Science": ["Python", "Data Structures", "Algorithms", "Machine Learning", "Web Development"],
    "History": ["Ancient History", "Modern History", "World Wars", "Indian History", "Political Science"],
    "Literature": ["English Literature", "Poetry", "Creative Writing", "Grammar", "Comprehension"],
}

DIFFICULTY_LEVELS = ["Beginner", "Intermediate", "Advanced"]

def init_state():
    defaults = {
        "student_name": "",
        "selected_subject": "Mathematics",
        "selected_topic": "Algebra",
        "difficulty": "Beginner",
        "chat_history": [],
        "quiz_questions": [],
        "quiz_answers": {},
        "quiz_submitted": False,
        "quiz_score": 0,
        "progress": {},
        "learning_path": [],
        "total_sessions": 0,
        "correct_answers": 0,
        "total_questions": 0,
        "page": "home",
        "concepts_learned": [],
        "weak_areas": [],
        "strong_areas": [],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()


def _groq_post(messages: list, max_tokens: int = 1500) -> str:
    """Core Groq API call (OpenAI-compatible format)."""
    try:
        resp = requests.post(
            GROQ_API_URL,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {GROQ_API_KEY}",
            },
            json={
                "model": MODEL,
                "max_tokens": max_tokens,
                "messages": messages,
            },
            timeout=30,
        )
        data = resp.json()
        if "choices" in data and data["choices"]:
            return data["choices"][0]["message"]["content"]
        # Surface any error message from Groq
        return f"⚠️ {data.get('error', {}).get('message', 'No response received.')}"
    except Exception as e:
        return f"⚠️ API Error: {str(e)}"


def call_claude(system_prompt: str, user_message: str, max_tokens: int = 1500) -> str:
    """Single-turn call via Groq."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_message},
    ]
    return _groq_post(messages, max_tokens)


def call_claude_with_history(system_prompt: str, messages: list, max_tokens: int = 1500) -> str:
    """Multi-turn call with conversation history via Groq."""
    full_messages = [{"role": "system", "content": system_prompt}] + messages
    return _groq_post(full_messages, max_tokens)


def explainer_agent(topic: str, subject: str, difficulty: str, question: str) -> str:
    system = f"""You are an expert, enthusiastic tutor specializing in {subject}.
Your role is to explain concepts clearly and engagingly at a {difficulty} level.
Current topic: {topic}

Guidelines:
- Use simple analogies and real-world examples
- Break complex ideas into digestible steps
- Use emojis sparingly to make content friendly
- Format with clear headings and bullet points where helpful
- Encourage curiosity and questions
- Adapt explanation depth to {difficulty} level
- Keep responses focused and not too long (300-400 words max unless more depth is asked)"""

    history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.chat_history[-8:]]
    history.append({"role": "user", "content": question})
    return call_claude_with_history(system, history)



def quiz_generator_agent(topic: str, subject: str, difficulty: str, num_questions: int = 5) -> list:
    system = """You are a quiz designer. Generate multiple-choice questions strictly as JSON.
Return ONLY a JSON array, no markdown, no extra text."""

    prompt = f"""Create {num_questions} multiple-choice quiz questions about "{topic}" in {subject} at {difficulty} level.

Return ONLY this JSON format (no markdown, no explanation):
[
  {{
    "question": "Question text here?",
    "options": ["A) Option 1", "B) Option 2", "C) Option 3", "D) Option 4"],
    "answer": "A) Option 1",
    "explanation": "Brief explanation of why this is correct."
  }}
]"""

    raw = call_claude(system, prompt, max_tokens=2000)
    try:
        clean = raw.strip().replace("```json", "").replace("```", "").strip()
        questions = json.loads(clean)
        return questions if isinstance(questions, list) else []
    except Exception:
        try:
            start = raw.find("[")
            end = raw.rfind("]") + 1
            if start != -1 and end > start:
                return json.loads(raw[start:end])
        except Exception:
            pass
        return []


def progress_tracker_agent(student_data: dict) -> str:
    system = """You are a learning analytics expert who gives personalized, motivating feedback.
Analyze student performance and provide actionable insights."""

    prompt = f"""Analyze this student's learning data and provide personalized feedback:

Student: {student_data.get('name', 'Student')}
Subject: {student_data.get('subject')}
Topic: {student_data.get('topic')}
Difficulty Level: {student_data.get('difficulty')}
Quiz Score: {student_data.get('score', 0)}/{student_data.get('total', 0)} ({student_data.get('pct', 0):.0f}%)
Concepts Learned: {', '.join(student_data.get('concepts', [])) or 'None recorded'}
Weak Areas: {', '.join(student_data.get('weak', [])) or 'None identified'}
Strong Areas: {', '.join(student_data.get('strong', [])) or 'None identified'}
Total Sessions: {student_data.get('sessions', 0)}
Overall Accuracy: {student_data.get('accuracy', 0):.0f}%

Provide:
1. 🎯 **Performance Summary** (2-3 sentences)
2. 💪 **Strengths** (bullet points)
3. 🔧 **Areas to Improve** (bullet points)
4. 📚 **Personalized Next Steps** (3-4 specific recommendations)
5. 🗺️ **Suggested Learning Path** (3 topics to tackle next in order)

Be encouraging, specific, and actionable."""

    return call_claude(system, prompt)


def render_sidebar():
    with st.sidebar:
        st.markdown('<div class="sidebar-logo">🧠 EduAI Tutor</div>', unsafe_allow_html=True)
        st.markdown("---")

        # Student name
        name = st.text_input("👤 Your Name", value=st.session_state.student_name, placeholder="Enter your name")
        if name != st.session_state.student_name:
            st.session_state.student_name = name

        st.markdown("### 📚 Learning Settings")

        subject = st.selectbox("Subject", list(SUBJECTS.keys()),
                               index=list(SUBJECTS.keys()).index(st.session_state.selected_subject))
        if subject != st.session_state.selected_subject:
            st.session_state.selected_subject = subject
            st.session_state.selected_topic = SUBJECTS[subject][0]

        topic = st.selectbox("Topic", SUBJECTS[st.session_state.selected_subject],
                             index=SUBJECTS[st.session_state.selected_subject].index(st.session_state.selected_topic)
                             if st.session_state.selected_topic in SUBJECTS[st.session_state.selected_subject] else 0)
        st.session_state.selected_topic = topic

        difficulty = st.select_slider("Difficulty", DIFFICULTY_LEVELS,
                                      value=st.session_state.difficulty)
        st.session_state.difficulty = difficulty

        st.markdown("---")
        st.markdown("### 🧭 Navigation")

        pages = {"🏠 Home": "home", "💬 AI Tutor": "tutor", "📝 Quiz": "quiz", "📊 Progress": "progress"}
        for label, page_key in pages.items():
            active = "nav-active" if st.session_state.page == page_key else ""
            if st.button(label, key=f"nav_{page_key}", use_container_width=True):
                st.session_state.page = page_key
                st.rerun()

        st.markdown("---")
        # Quick stats
        acc = (st.session_state.correct_answers / max(st.session_state.total_questions, 1)) * 100
        st.markdown(f"""
        <div class="stat-card-small">
            <div>📅 Sessions: <b>{st.session_state.total_sessions}</b></div>
            <div>✅ Accuracy: <b>{acc:.0f}%</b></div>
            <div>❓ Questions: <b>{st.session_state.total_questions}</b></div>
        </div>
        """, unsafe_allow_html=True)


def render_home():
    st.markdown("""
    <div class="hero-section">
        <div class="hero-title">🧠 EduAI Tutor</div>
        <div class="hero-desc">Powered by a multi-agent AI system — one tutor explains,<br>one creates quizzes, and one tracks your personalized progress.</div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">💬</div>
            <div class="feature-title">AI Tutor Agent</div>
            <div class="feature-desc">Chat with an expert AI tutor that explains any concept in depth, answers follow-up questions, and adapts to your level.</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">📝</div>
            <div class="feature-title">Quiz Generator Agent</div>
            <div class="feature-desc">Auto-generates tailored MCQ quizzes on your chosen topic with instant feedback and detailed explanations.</div>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">📊</div>
            <div class="feature-title">Progress Tracker Agent</div>
            <div class="feature-desc">Analyzes your performance using NLP clustering to identify weak areas and build a personalized learning path.</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 🚀 Quick Start")
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("💬 Start Tutoring Session", use_container_width=True):
            st.session_state.page = "tutor"
            st.rerun()
    with c2:
        if st.button("📝 Take a Quiz", use_container_width=True):
            st.session_state.page = "quiz"
            st.rerun()
    with c3:
        if st.button("📊 View My Progress", use_container_width=True):
            st.session_state.page = "progress"
            st.rerun()

    # Architecture diagram
   # st.markdown("---")
   # st.markdown("### 🏗️ Multi-Agent Architecture")
   # st.markdown("""
   # <div class="arch-box">
    #    <div class="arch-row">
     #       <div class="arch-node student">👤 Student Input</div>
      #      <div class="arch-arrow">→</div>
       #     <div class="arch-node router">🔀 Orchestrator</div>
        #</div>
        #<div class="arch-agents">
         #   <div class="arch-node agent1">🎓 Explainer Agent<br><small>Concept explanation & Q&A</small></div>
          #  <div class="arch-node agent2">📝 Quiz Agent<br><small>NLP question generation</small></div>
           # <div class="arch-node agent3">📊 Tracker Agent<br><small>Clustering & personalization</small></div>
        #</div>
        #<div class="arch-row">
         #   <div class="arch-node output">🎯 Adaptive Learning Path</div>
        #</div>
    #</div>
    #""", unsafe_allow_html=True)


def render_tutor():
    st.markdown(f'<div class="page-header">💬 AI Tutor — {st.session_state.selected_topic}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="page-sub">{st.session_state.selected_subject} · {st.session_state.difficulty} Level</div>', unsafe_allow_html=True)

    # Starter prompts
    if not st.session_state.chat_history:
        st.markdown("#### 💡 Try asking:")
        starter_cols = st.columns(3)
        starters = [
            f"Explain {st.session_state.selected_topic} from scratch",
            f"Give me a real-world example of {st.session_state.selected_topic}",
            f"What are the key formulas in {st.session_state.selected_topic}?",
        ]
        for i, s in enumerate(starters):
            with starter_cols[i]:
                if st.button(s, key=f"starter_{i}", use_container_width=True):
                    st.session_state.chat_history.append({"role": "user", "content": s})
                    with st.spinner("🤔 Thinking..."):
                        reply = explainer_agent(
                            st.session_state.selected_topic,
                            st.session_state.selected_subject,
                            st.session_state.difficulty,
                            s
                        )
                    st.session_state.chat_history.append({"role": "assistant", "content": reply})
                    st.session_state.total_sessions += 1
                    if st.session_state.selected_topic not in st.session_state.concepts_learned:
                        st.session_state.concepts_learned.append(st.session_state.selected_topic)
                    st.rerun()

    # Chat history
    chat_container = st.container()
    with chat_container:
        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                st.markdown(f"""
                <div class="chat-bubble user-bubble">
                    <span class="bubble-icon">👤</span>
                    <div class="bubble-content">{msg['content']}</div>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="chat-bubble ai-bubble">
                    <span class="bubble-icon">🧠</span>
                    <div class="bubble-content">{msg['content']}</div>
                </div>""", unsafe_allow_html=True)

    # Input
    st.markdown("<br>", unsafe_allow_html=True)
    with st.form("chat_form", clear_on_submit=True):
        col_in, col_btn = st.columns([5, 1])
        with col_in:
            user_input = st.text_input("Ask anything...", placeholder=f"Ask about {st.session_state.selected_topic}...", label_visibility="collapsed")
        with col_btn:
            submitted = st.form_submit_button("Send 🚀", use_container_width=True)

    if submitted and user_input.strip():
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.spinner("🤔 Thinking..."):
            reply = explainer_agent(
                st.session_state.selected_topic,
                st.session_state.selected_subject,
                st.session_state.difficulty,
                user_input
            )
        st.session_state.chat_history.append({"role": "assistant", "content": reply})
        st.session_state.total_sessions += 1
        if st.session_state.selected_topic not in st.session_state.concepts_learned:
            st.session_state.concepts_learned.append(st.session_state.selected_topic)
        st.rerun()

    if st.session_state.chat_history:
        if st.button("🗑️ Clear Chat"):
            st.session_state.chat_history = []
            st.rerun()


def render_quiz():
    st.markdown(f'<div class="page-header">📝 Quiz — {st.session_state.selected_topic}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="page-sub">{st.session_state.selected_subject} · {st.session_state.difficulty} Level</div>', unsafe_allow_html=True)

    col_gen, col_num = st.columns([3, 1])
    with col_num:
        num_q = st.selectbox("# Questions", [3, 5, 10], index=1)
    with col_gen:
        if st.button("🎲 Generate New Quiz", use_container_width=True):
            with st.spinner("🤖 Quiz Agent is crafting questions..."):
                questions = quiz_generator_agent(
                    st.session_state.selected_topic,
                    st.session_state.selected_subject,
                    st.session_state.difficulty,
                    num_q
                )
            if questions:
                st.session_state.quiz_questions = questions
                st.session_state.quiz_answers = {}
                st.session_state.quiz_submitted = False
                st.session_state.quiz_score = 0
                st.rerun()
            else:
                st.error("Failed to generate quiz. Please try again.")

    if not st.session_state.quiz_questions:
        st.markdown("""
        <div class="empty-state">
            <div style="font-size:3rem">📝</div>
            <div>Click <b>Generate New Quiz</b> to start!</div>
            <div style="font-size:0.85rem;opacity:0.7;margin-top:0.5rem">The Quiz Agent will create personalized questions for your topic.</div>
        </div>""", unsafe_allow_html=True)
        return

    # Render questions
    for i, q in enumerate(st.session_state.quiz_questions):
        answered = st.session_state.quiz_answers.get(i)
        correct = answered == q.get("answer") if answered else None

        card_class = "quiz-card"
        if st.session_state.quiz_submitted:
            card_class += " correct-card" if correct else " wrong-card"

        st.markdown(f'<div class="{card_class}">', unsafe_allow_html=True)
        st.markdown(f"**Q{i+1}. {q['question']}**")

        options = q.get("options", [])
        if not st.session_state.quiz_submitted:
            chosen = st.radio("", options, key=f"q_{i}", index=None, label_visibility="collapsed")
            if chosen:
                st.session_state.quiz_answers[i] = chosen
        else:
            for opt in options:
                if opt == q.get("answer"):
                    st.markdown(f"✅ **{opt}**")
                elif opt == st.session_state.quiz_answers.get(i):
                    st.markdown(f"❌ ~~{opt}~~")
                else:
                    st.markdown(f"　{opt}")
            if st.session_state.quiz_submitted:
                st.info(f"💡 {q.get('explanation', '')}")

        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    if not st.session_state.quiz_submitted:
        answered_count = len(st.session_state.quiz_answers)
        total = len(st.session_state.quiz_questions)
        st.markdown(f"*Answered: {answered_count}/{total}*")

        if st.button("✅ Submit Quiz", use_container_width=True, disabled=answered_count == 0):
            score = sum(
                1 for i, q in enumerate(st.session_state.quiz_questions)
                if st.session_state.quiz_answers.get(i) == q.get("answer")
            )
            st.session_state.quiz_submitted = True
            st.session_state.quiz_score = score
            st.session_state.correct_answers += score
            st.session_state.total_questions += total

            # Track weak/strong areas
            pct = score / total * 100
            topic = st.session_state.selected_topic
            if pct >= 70:
                if topic not in st.session_state.strong_areas:
                    st.session_state.strong_areas.append(topic)
            else:
                if topic not in st.session_state.weak_areas:
                    st.session_state.weak_areas.append(topic)

            # Update progress dict
            st.session_state.progress[topic] = {
                "score": score, "total": total, "pct": pct,
                "difficulty": st.session_state.difficulty,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M")
            }
            st.rerun()
    else:
        score = st.session_state.quiz_score
        total = len(st.session_state.quiz_questions)
        pct = score / total * 100
        emoji = "🏆" if pct >= 80 else "👍" if pct >= 60 else "📖"
        st.markdown(f"""
        <div class="score-banner">
            {emoji} Score: <b>{score}/{total}</b> ({pct:.0f}%)
        </div>""", unsafe_allow_html=True)

        if st.button("🔁 Retake / New Quiz", use_container_width=True):
            st.session_state.quiz_questions = []
            st.session_state.quiz_answers = {}
            st.session_state.quiz_submitted = False
            st.rerun()



def render_progress():
    st.markdown('<div class="page-header">📊 My Learning Progress</div>', unsafe_allow_html=True)
    name = st.session_state.student_name or "Student"

    # KPI cards
    acc = (st.session_state.correct_answers / max(st.session_state.total_questions, 1)) * 100
    k1, k2, k3, k4 = st.columns(4)
    kpis = [
        ("📅", "Sessions", str(st.session_state.total_sessions)),
        ("❓", "Questions", str(st.session_state.total_questions)),
        ("✅", "Accuracy", f"{acc:.0f}%"),
        ("📚", "Topics", str(len(st.session_state.concepts_learned))),
    ]
    for col, (icon, label, val) in zip([k1, k2, k3, k4], kpis):
        with col:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-icon">{icon}</div>
                <div class="kpi-val">{val}</div>
                <div class="kpi-label">{label}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Topic performance
    if st.session_state.progress:
        st.markdown("### 📈 Quiz Performance by Topic")
        for topic, data in st.session_state.progress.items():
            pct = data["pct"]
            bar_color = "#10b981" if pct >= 70 else "#f59e0b" if pct >= 50 else "#ef4444"
            st.markdown(f"""
            <div class="progress-row">
                <div class="progress-label">{topic}</div>
                <div class="progress-bar-bg">
                    <div class="progress-bar-fill" style="width:{pct}%;background:{bar_color};"></div>
                </div>
                <div class="progress-pct">{pct:.0f}%</div>
                <div class="progress-meta">{data['score']}/{data['total']} · {data['difficulty']}</div>
            </div>""", unsafe_allow_html=True)

    col_strong, col_weak = st.columns(2)
    with col_strong:
        st.markdown("### 💪 Strong Areas")
        if st.session_state.strong_areas:
            for a in st.session_state.strong_areas:
                st.markdown(f"<div class='area-tag strong-tag'>✅ {a}</div>", unsafe_allow_html=True)
        else:
            st.info("Take quizzes to identify your strengths!")

    with col_weak:
        st.markdown("### 🔧 Areas to Improve")
        if st.session_state.weak_areas:
            for a in st.session_state.weak_areas:
                st.markdown(f"<div class='area-tag weak-tag'>⚠️ {a}</div>", unsafe_allow_html=True)
        else:
            st.info("No weak areas identified yet.")

    # AI Feedback
    st.markdown("---")
    st.markdown("### 🤖 AI Progress Analysis")
    if st.button("🔍 Get Personalized Feedback & Learning Path", use_container_width=True):
        if st.session_state.total_questions == 0:
            st.warning("Complete at least one quiz to get personalized feedback!")
        else:
            student_data = {
                "name": name,
                "subject": st.session_state.selected_subject,
                "topic": st.session_state.selected_topic,
                "difficulty": st.session_state.difficulty,
                "score": st.session_state.quiz_score,
                "total": len(st.session_state.quiz_questions) or 1,
                "pct": acc,
                "concepts": st.session_state.concepts_learned,
                "weak": st.session_state.weak_areas,
                "strong": st.session_state.strong_areas,
                "sessions": st.session_state.total_sessions,
                "accuracy": acc,
            }
            with st.spinner("🤖 Progress Tracker Agent is analyzing your journey..."):
                feedback = progress_tracker_agent(student_data)
            st.markdown(f"""
            <div class="feedback-box">
                {feedback}
            </div>""", unsafe_allow_html=True)


render_sidebar()

if st.session_state.page == "home":
    render_home()
elif st.session_state.page == "tutor":
    render_tutor()
elif st.session_state.page == "quiz":
    render_quiz()
elif st.session_state.page == "progress":
    render_progress()
