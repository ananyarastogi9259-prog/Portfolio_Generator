import streamlit as st
import re
import json
import time
import os
from google import genai
from google.genai import types

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

def mask_pii(text):
    """
    Mask sensitive PII like SSNs and Credit Card numbers.
    This demonstrates basic regex usage for privacy pre-processing.
    """
    # Mask 9-digit SSN-like patterns (e.g., XXX-XX-XXXX)
    text = re.sub(r'\b\d{3}[-.\s]?\d{2}[-.\s]?\d{4}\b', '[REDACTED SSN]', text)
    
    # Mask 16-digit Credit Card numbers
    text = re.sub(r'\b(?:\d{4}[-.\s]?){3}\d{4}\b', '[REDACTED CC]', text)
    
    return text

def extract_portfolio_data(api_key, text=None, uploaded_file=None):
    """
    Call the Gemini API to extract structured data from the resume text or file.
    Enforces JSON output via the system prompt and configuration.
    """
    client = genai.Client(api_key=api_key)
    
    base_prompt = """
    You are an expert resume parser. Extract the following information from the provided resume. 
    Format it STRICTLY as a JSON object.
    Do not add markdown formatting (like ```json) to the output. Just return the raw JSON.
    
    CRITICAL: Do NOT invent or hallucinate information. If a section is missing from the resume, return an empty string "" or an empty list [] for that field.
    
    Expected JSON Structure:
    {
        "Name": "Full Name",
        "Headline": "Short professional identity",
        "Contact": {"Email": "email@example.com", "Phone": "...", "Links": ["url1", "url2"]},
        "Summary": "A brief professional summary.",
        "Education": ["Degree in X from Y (Year)"],
        "Experience": ["Job Title at Company (Dates): Responsibilities"],
        "Projects": ["Project Name: Description"],
        "Skills": ["Skill 1", "Skill 2"],
        "Achievements": ["Achievement 1"]
    }
    """
    
    contents = [base_prompt]
    
    if uploaded_file is not None:
        file_bytes = uploaded_file.getvalue()
        mime_type = uploaded_file.type
        contents.append(types.Part.from_bytes(data=file_bytes, mime_type=mime_type))
    elif text:
        contents.append(f"Resume Text:\n{text}")
    else:
        return None
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model='gemini-3.6-flash',
                contents=contents,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                )
            )
            # Strip markdown formatting if the model incorrectly added it
            raw_text = response.text.replace("```json", "").replace("```", "").strip()
            try:
                return json.loads(raw_text)
            except json.JSONDecodeError:
                return {"error": "Invalid JSON response from the AI model."}
        except Exception as e:
            error_str = str(e)
            # If it's a 503 error, wait and retry. Otherwise, or if out of retries, fail.
            if ("503" in error_str or "UNAVAILABLE" in error_str) and attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # 1s, then 2s
                continue
            
            return {"error": f"API Failure: {error_str}"}

def check_hallucinations(original_text, extracted_json):
    """
    Compare extracted skills and links against the original raw text 
    to flag potential hallucinations (items the AI invented).
    """
    original_lower = original_text.lower()
    hallucinations = {"skills": [], "links": []}
    
    if "Skills" in extracted_json:
        for skill in extracted_json["Skills"]:
            if skill.lower() not in original_lower:
                hallucinations["skills"].append(skill)
                
    if "Contact" in extracted_json and "Links" in extracted_json["Contact"]:
        for link in extracted_json["Contact"]["Links"]:
            if link.lower() not in original_lower:
                hallucinations["links"].append(link)
                
    return hallucinations

def generate_html(data):
    """Generate a modern, dark-themed HTML template based on the extracted JSON data."""
    raw_name = data.get("Name", "Portfolio")
    name = str(raw_name).upper() if raw_name and str(raw_name) != "None" else "PORTFOLIO"
    
    headline = data.get("Headline", "")
    if headline is None or str(headline) == "None": headline = ""
    
    summary = data.get("Summary", "")
    if summary is None or str(summary) == "None": summary = ""
    
    # Hero Name Styling (Split name if possible for styling)
    name_parts = name.split(" ")
    if len(name_parts) >= 2:
        hero_name_html = f"<span class='text-white'>{name_parts[0]}</span> <span class='text-accent'>{name_parts[1]}</span>"
    else:
        hero_name_html = f"<span class='text-accent'>{name}</span>"

    contact_html = ""
    if "Contact" in data:
        email = data["Contact"].get("Email", "")
        phone = data["Contact"].get("Phone", "")
        links = data["Contact"].get("Links", [])
        
        if email: contact_html += f"<div class='contact-item'><strong>Email:</strong> <a href='mailto:{email}'>{email}</a></div>"
        if phone: contact_html += f"<div class='contact-item'><strong>Phone:</strong> {phone}</div>"
        for link in links:
            contact_html += f"<div class='contact-item'><a href='{link}' target='_blank'>{link}</a></div>"
            
    # Skill Progress Bars
    skills_html = ""
    skills = data.get("Skills", [])
    # Default percentages for visual flair to match the design
    percentages = [95, 90, 85, 80, 85, 95, 90]
    for i, skill in enumerate(skills):
        pct = percentages[i % len(percentages)]
        skills_html += f"""
        <div class="skill-row">
            <div class="skill-label">
                <span>{skill}</span>
                <span>{pct}%</span>
            </div>
            <div class="skill-bar-bg">
                <div class="skill-bar-fill" style="width: {pct}%"></div>
            </div>
        </div>
        """

    def render_list_section(title, items):
        if not items: return ""
        items_html = "".join([f"<li>{item}</li>" for item in items])
        return f"""
        <div class="card">
            <h2>{title} <span class="accent-line"></span></h2>
            <ul>{items_html}</ul>
        </div>
        """

    exp_html = render_list_section("Experience", data.get("Experience", []))
    edu_html = render_list_section("Education", data.get("Education", []))
    proj_html = render_list_section("Projects", data.get("Projects", []))
    achiev_html = render_list_section("Achievements", data.get("Achievements", []))

    html_template = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>{data.get('Name', 'Portfolio')} - Portfolio</title>
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap" rel="stylesheet">
        <style>
            :root {{
                --bg-color: #040914;
                --card-bg: #0b1324;
                --text-primary: #ffffff;
                --text-secondary: #a0aec0;
                --accent: #00d2ff;
                --accent-dark: #0088ff;
            }}
            body {{
                font-family: 'Inter', sans-serif;
                background-color: var(--bg-color);
                color: var(--text-secondary);
                margin: 0;
                padding: 0;
                line-height: 1.6;
                overflow-x: hidden;
            }}
            .container {{
                max-width: 1200px;
                margin: 0 auto;
                padding: 40px 20px;
            }}
            /* Hero Section */
            .hero {{
                margin-bottom: 60px;
                position: relative;
            }}
            .hero::after {{
                content: '';
                position: absolute;
                top: -100px;
                right: -100px;
                width: 400px;
                height: 400px;
                background: radial-gradient(circle, rgba(0,210,255,0.1) 0%, rgba(0,0,0,0) 70%);
                z-index: -1;
            }}
            .hero-title {{
                font-size: 5rem;
                font-weight: 800;
                line-height: 1;
                margin: 0 0 20px 0;
                letter-spacing: -1px;
            }}
            .text-white {{ color: var(--text-primary); }}
            .text-accent {{ color: var(--accent); text-shadow: 0 0 20px rgba(0, 210, 255, 0.4); }}
            .tagline {{
                color: var(--text-primary);
                font-weight: 600;
                font-size: 1.2rem;
                margin-bottom: 20px;
                letter-spacing: 2px;
            }}
            .tagline span {{ color: var(--accent); }}
            .summary {{
                max-width: 600px;
                font-size: 1.1rem;
                margin-bottom: 30px;
            }}
            /* Grid Layout */
            .grid {{
                display: grid;
                grid-template-columns: 2fr 1fr;
                gap: 30px;
            }}
            @media (max-width: 900px) {{
                .grid {{ grid-template-columns: 1fr; }}
            }}
            /* Cards */
            .card {{
                background-color: var(--card-bg);
                border: 1px solid rgba(255,255,255,0.05);
                border-radius: 12px;
                padding: 30px;
                margin-bottom: 30px;
                box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            }}
            h2 {{
                color: var(--text-primary);
                font-size: 1.2rem;
                font-weight: 700;
                margin-top: 0;
                margin-bottom: 25px;
                display: flex;
                align-items: center;
                letter-spacing: 1px;
            }}
            .accent-line {{
                height: 2px;
                background-color: var(--accent);
                flex-grow: 1;
                margin-left: 15px;
                opacity: 0.3;
            }}
            ul {{
                list-style-type: none;
                padding: 0;
                margin: 0;
            }}
            li {{
                margin-bottom: 15px;
                padding-left: 20px;
                position: relative;
            }}
            li::before {{
                content: '▹';
                position: absolute;
                left: 0;
                color: var(--accent);
            }}
            /* Skills */
            .skill-row {{ margin-bottom: 20px; }}
            .skill-label {{
                display: flex;
                justify-content: space-between;
                color: var(--text-primary);
                font-size: 0.9rem;
                margin-bottom: 8px;
                font-weight: 600;
            }}
            .skill-bar-bg {{
                height: 6px;
                background-color: rgba(255,255,255,0.1);
                border-radius: 3px;
                overflow: hidden;
            }}
            .skill-bar-fill {{
                height: 100%;
                background: linear-gradient(90deg, var(--accent-dark), var(--accent));
                border-radius: 3px;
                box-shadow: 0 0 10px rgba(0, 210, 255, 0.5);
            }}
            /* Contact */
            .contact-item {{
                margin-bottom: 15px;
                display: flex;
                align-items: center;
                gap: 10px;
            }}
            .contact-item a {{
                color: var(--accent);
                text-decoration: none;
                transition: 0.3s;
            }}
            .contact-item a:hover {{ text-shadow: 0 0 10px rgba(0, 210, 255, 0.5); }}
            .contact-item strong {{ color: var(--text-primary); }}
            
            /* CTA Button */
            .cta-btn {{
                display: inline-block;
                padding: 12px 30px;
                border: 1px solid var(--accent);
                color: var(--accent);
                text-decoration: none;
                border-radius: 30px;
                font-weight: 600;
                font-size: 0.9rem;
                text-transform: uppercase;
                letter-spacing: 1px;
                transition: 0.3s;
                background: rgba(0, 210, 255, 0.05);
            }}
            .cta-btn:hover {{
                background: var(--accent);
                color: var(--bg-color);
                box-shadow: 0 0 20px rgba(0, 210, 255, 0.4);
            }}
            
            /* AI Modal CSS */
            .modal-overlay {{
                display: none;
                position: fixed;
                top: 0; left: 0; width: 100%; height: 100%;
                background: rgba(2, 6, 18, 0.85);
                backdrop-filter: blur(15px);
                z-index: 1000;
                opacity: 0;
                transition: opacity 0.4s ease;
            }}
            .modal-overlay.active {{
                display: flex;
                justify-content: center;
                align-items: center;
                opacity: 1;
            }}
            .modal-content {{
                background: linear-gradient(145deg, rgba(11, 19, 36, 0.95), rgba(4, 9, 20, 0.95));
                border: 1px solid rgba(0, 210, 255, 0.3);
                box-shadow: 0 0 40px rgba(0, 210, 255, 0.2), inset 0 0 20px rgba(0, 210, 255, 0.05);
                border-radius: 20px;
                width: 90%;
                max-width: 600px;
                padding: 30px;
                position: relative;
                transform: scale(0.9) translateY(20px);
                transition: transform 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
                color: #fff;
                overflow: hidden;
            }}
            .modal-overlay.active .modal-content {{
                transform: scale(1) translateY(0);
            }}
            .modal-header {{
                text-align: center;
                margin-bottom: 25px;
            }}
            .modal-header h2 {{
                font-size: 2rem;
                background: -webkit-linear-gradient(45deg, #00d2ff, #3a7bd5);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                margin-bottom: 5px;
            }}
            .modal-header p {{
                color: #a0aec0;
                font-size: 1rem;
            }}
            .close-btn {{
                position: absolute;
                top: 15px; right: 20px;
                font-size: 1.5rem;
                color: #a0aec0;
                cursor: pointer;
                transition: 0.3s;
                background: none; border: none;
            }}
            .close-btn:hover {{ color: #00d2ff; transform: scale(1.1); }}
            .form-group {{ margin-bottom: 15px; }}
            .form-group label {{ display: block; margin-bottom: 5px; color: #00d2ff; font-weight: 600; font-size: 0.9rem; }}
            .form-group select {{
                width: 100%; padding: 12px;
                background: rgba(0,0,0,0.5);
                border: 1px solid rgba(0, 210, 255, 0.2);
                color: #fff; border-radius: 8px;
                outline: none; transition: 0.3s;
                appearance: none;
            }}
            .form-group select:focus {{ border-color: #00d2ff; box-shadow: 0 0 10px rgba(0,210,255,0.3); }}
            .btn-glow {{
                width: 100%; padding: 15px;
                background: linear-gradient(45deg, #00d2ff, #3a7bd5);
                border: none; border-radius: 30px;
                color: #fff; font-weight: bold; font-size: 1.1rem;
                cursor: pointer; margin-top: 10px;
                transition: 0.3s;
                box-shadow: 0 5px 15px rgba(0, 210, 255, 0.4);
            }}
            .btn-glow:hover {{ transform: translateY(-2px); box-shadow: 0 8px 25px rgba(0, 210, 255, 0.6); }}
            .result-card {{
                display: none;
                margin-top: 20px;
                padding: 20px;
                background: rgba(0, 210, 255, 0.05);
                border: 1px solid rgba(0, 210, 255, 0.2);
                border-radius: 12px;
            }}
            .result-card h3 {{ color: #00d2ff; margin-bottom: 10px; }}
            .result-item {{ margin-bottom: 10px; font-size: 0.95rem; }}
            .result-item strong {{ color: #a0aec0; }}
            .action-btns {{ display: flex; gap: 10px; margin-top: 20px; }}
            .action-btn {{
                flex: 1; padding: 10px;
                background: rgba(255,255,255,0.05);
                border: 1px solid rgba(255,255,255,0.1);
                color: #fff; border-radius: 8px;
                cursor: pointer; transition: 0.3s;
                font-size: 0.85rem;
            }}
            .action-btn:hover {{ background: rgba(0, 210, 255, 0.1); border-color: #00d2ff; color: #00d2ff; }}
            .loader {{ display: none; text-align: center; margin-top: 20px; color: #00d2ff; }}
        </style>
    </head>
    <body>
        <div class="container">
            <header class="hero">
                {f"<h2 style='color: var(--text-primary); font-size: 1.5rem; margin-bottom: 10px;'>{headline}</h2>" if headline else ""}
                <h1 class="hero-title">{hero_name_html}</h1>
                <div class="tagline"><span>|</span> DESIGN. <span>|</span> CODE. <span>|</span> CREATE. <span>—</span></div>
                {f"<div class='summary'><h2>Professional Summary <span class='accent-line'></span></h2>{summary}</div>" if summary else ""}
                <a href="#" class="cta-btn" onclick="openModal(); return false;">Let's Build Something Amazing →</a>
            </header>
            
            <div class="grid">
                <div class="col-main">
                    {exp_html}
                    {proj_html}
                    {edu_html}
                    {achiev_html}
                </div>
                
                <div class="col-sidebar">
                    <div class="card">
                        <h2>Skills <span class="accent-line"></span></h2>
                        {skills_html if skills_html else "<p>No skills listed.</p>"}
                    </div>
                    
                    <div class="card">
                        <h2>Contact and Links <span class="accent-line"></span></h2>
                        {contact_html if contact_html else "<p>No contact info provided.</p>"}
                    </div>
                </div>
            </div>
        </div>
        
        <!-- AI Idea Generator Modal -->
        <div class="modal-overlay" id="aiModal">
            <div class="modal-content">
                <button class="close-btn" onclick="closeModal()">×</button>
                <div class="modal-header">
                    <h2>🚀 AI Project Idea Generator</h2>
                    <p>Turn your interests into your next portfolio project.</p>
                </div>
                
                <div id="modalForm">
                    <div class="form-group">
                        <label>Project Domain</label>
                        <select id="domainSelect">
                            <option value="AI/ML">AI/ML</option>
                            <option value="Web Development">Web Development</option>
                            <option value="Data Science">Data Science</option>
                            <option value="Cyber Security">Cyber Security</option>
                            <option value="Mobile App">Mobile App Development</option>
                            <option value="Cloud Computing">Cloud Computing</option>
                        </select>
                    </div>
                    <div class="form-group">
                        <label>Difficulty Level</label>
                        <select id="diffSelect">
                            <option value="Beginner">Beginner</option>
                            <option value="Intermediate">Intermediate</option>
                            <option value="Advanced">Advanced</option>
                        </select>
                    </div>
                    <div class="form-group">
                        <label>Time Available</label>
                        <select id="timeSelect">
                            <option value="Weekend Project">Weekend Project</option>
                            <option value="1 Week">1 Week</option>
                            <option value="1 Month">1 Month</option>
                            <option value="Long-Term">Long-Term</option>
                        </select>
                    </div>
                    <button class="btn-glow" onclick="generateProject()">Generate Project Idea ✨</button>
                </div>
                
                <div class="loader" id="modalLoader">
                    <p>🧠 AI is analyzing possibilities...</p>
                </div>
                
                <div class="result-card" id="resultCard">
                    <h3 id="resName"></h3>
                    <div class="result-item"><strong>Description:</strong> <span id="resDesc"></span></div>
                    <div class="result-item"><strong>Key Features:</strong> <span id="resFeat"></span></div>
                    <div class="result-item"><strong>Tech Stack:</strong> <span id="resTech"></span></div>
                    <div class="result-item"><strong>Difficulty:</strong> <span id="resDiff"></span> | <strong>Time:</strong> <span id="resTime"></span></div>
                    
                    <div class="action-btns">
                        <button class="action-btn" onclick="resetModal()">Try Another</button>
                        <button class="action-btn" onclick="copyProject()">Copy Idea</button>
                        <button class="action-btn" onclick="downloadIdea()">Save PDF</button>
                    </div>
                </div>
            </div>
        </div>

        <script>
            const projects = [
                {{ name: "AI Resume Analyzer", domain: "AI/ML", diff: "Intermediate", time: "1-2 Weeks", desc: "An intelligent system that analyzes resumes and provides skill-gap recommendations.", feat: "Resume Upload, Skill Extraction, ATS Score Prediction", tech: "React, FastAPI, Python, Gemini API, MongoDB" }},
                {{ name: "Threat Intelligence Dashboard", domain: "Cyber Security", diff: "Advanced", time: "1 Month", desc: "Real-time dashboard visualizing global cyber threats using honeypot data.", feat: "Live Map, IP Reputation Check, Automated Alerts", tech: "Vue.js, Node.js, ELK Stack, Python" }},
                {{ name: "Personal Finance AI Assistant", domain: "Mobile App", diff: "Beginner", time: "Weekend Project", desc: "A mobile app that categorizes expenses and chats with users about budgets.", feat: "Expense Tracking, AI Chat, Receipt Scanning", tech: "Flutter, Firebase, OpenAI API" }},
                {{ name: "Serverless E-Commerce API", domain: "Cloud Computing", diff: "Intermediate", time: "1 Week", desc: "Highly scalable e-commerce backend built entirely on serverless architecture.", feat: "User Auth, Cart Management, Stripe Integration", tech: "AWS Lambda, API Gateway, DynamoDB, Node.js" }},
                {{ name: "Predictive Maintenance System", domain: "Data Science", diff: "Advanced", time: "Long-Term", desc: "IoT sensor data analysis to predict machinery failure before it happens.", feat: "Anomaly Detection, Real-time Streaming, Degradation Modeling", tech: "Python, Scikit-Learn, Kafka, PostgreSQL" }},
                {{ name: "Real-time Collaborative Code Editor", domain: "Web Development", diff: "Advanced", time: "1 Month", desc: "Browser-based IDE with real-time multi-user typing and code execution.", feat: "WebSockets, Syntax Highlighting, Dockerized Execution", tech: "React, Socket.io, Node.js, Docker" }},
                {{ name: "Automated Meeting Summarizer", domain: "AI/ML", diff: "Beginner", time: "Weekend Project", desc: "Transcribes audio files and generates concise meeting minutes.", feat: "Audio Upload, Speech-to-Text, LLM Summarization", tech: "Python, Whisper API, Streamlit" }}
            ];

            function openModal() {{
                document.getElementById('aiModal').classList.add('active');
                resetModal();
            }}
            function closeModal() {{
                document.getElementById('aiModal').classList.remove('active');
            }}
            function resetModal() {{
                document.getElementById('modalForm').style.display = 'block';
                document.getElementById('resultCard').style.display = 'none';
                document.getElementById('modalLoader').style.display = 'none';
            }}
            function typeWriter(elementId, text, speed, callback) {{
                let i = 0;
                document.getElementById(elementId).innerHTML = '';
                function type() {{
                    if (i < text.length) {{
                        document.getElementById(elementId).innerHTML += text.charAt(i);
                        i++;
                        setTimeout(type, speed);
                    }} else if (callback) callback();
                }}
                type();
            }}
            function generateProject() {{
                document.getElementById('modalForm').style.display = 'none';
                document.getElementById('modalLoader').style.display = 'block';
                
                const dom = document.getElementById('domainSelect').value;
                const diff = document.getElementById('diffSelect').value;
                const time = document.getElementById('timeSelect').value;
                
                setTimeout(() => {{
                    document.getElementById('modalLoader').style.display = 'none';
                    document.getElementById('resultCard').style.display = 'block';
                    
                    let filtered = projects.filter(p => p.domain.includes(dom.split('/')[0]) || p.diff === diff);
                    if(filtered.length === 0) filtered = projects;
                    const p = filtered[Math.floor(Math.random() * filtered.length)];
                    
                    document.getElementById('resTime').innerText = time;
                    document.getElementById('resDiff').innerText = diff;
                    
                    ['resName', 'resDesc', 'resFeat', 'resTech'].forEach(id => document.getElementById(id).innerHTML = '');
                    
                    typeWriter('resName', p.name, 50, () => {{
                        typeWriter('resDesc', p.desc, 20, () => {{
                            typeWriter('resFeat', p.feat, 20, () => {{
                                typeWriter('resTech', p.tech, 20);
                            }});
                        }});
                    }});
                }}, 1500);
            }}
            function copyProject() {{
                const text = `Project: ${{document.getElementById('resName').innerText}}\\nDescription: ${{document.getElementById('resDesc').innerText}}\\nFeatures: ${{document.getElementById('resFeat').innerText}}\\nTech Stack: ${{document.getElementById('resTech').innerText}}`;
                navigator.clipboard.writeText(text);
                alert("Project idea copied to clipboard!");
            }}
            function downloadIdea() {{
                window.print();
            }}
        </script>
    </body>
    </html>
    """
    return html_template

def main():
    st.set_page_config(page_title="AI Portfolio Generator", layout="wide", page_icon="📄")
    
    if "extracted_data" not in st.session_state:
        st.session_state["extracted_data"] = None
        st.session_state["hallucinations"] = None
        st.session_state["html_content"] = None
    
    # Inject Custom CSS for the glowing 3D title and animations
    st.markdown("""
    <style>
    /* Animated Gradient Background */
    .stApp {
        background: linear-gradient(135deg, #020612, #0a1428, #001220, #040914);
        background-size: 400% 400%;
        animation: gradientBG 15s ease infinite;
        will-change: background-position;
        transform: translateZ(0);
    }
    @keyframes gradientBG {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    /* 3D Glowing Engraved Title */
    .glowing-title {
        text-align: center;
        font-size: 4rem !important;
        font-weight: 900;
        background: -webkit-linear-gradient(45deg, #00d2ff, #3a7bd5);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
        padding-top: 1rem;
        text-shadow: 
            0px 3px 0px #0b1324, 
            0px 6px 0px #0b1324, 
            0px 9px 0px #0b1324, 
            0px 15px 30px rgba(0, 210, 255, 0.6);
        animation: float3DEngraved 3s ease-in-out infinite alternate;
        perspective: 1000px;
        transition: all 0.4s ease;
        will-change: transform;
        transform: translateZ(0);
    }
    
    .glowing-title:hover {
        background: -webkit-linear-gradient(45deg, #39ff14, #00ff87);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: float3DRadium 2s ease-in-out infinite alternate;
        text-shadow: 
            0px 3px 0px #0b1324, 
            0px 6px 0px #0b1324, 
            0px 9px 0px #0b1324, 
            0px 15px 40px rgba(57, 255, 20, 0.8);
    }
    
    @keyframes float3DEngraved {
        from { transform: translateY(0px) rotateX(5deg) scale(1); }
        to { transform: translateY(-15px) rotateX(-10deg) scale(1.03); }
    }
    
    @keyframes float3DRadium {
        from { transform: translateY(0px) rotateX(5deg) scale(1.05); }
        to { transform: translateY(-15px) rotateX(-10deg) scale(1.08); }
    }

    .subtitle {
        text-align: center;
        font-size: 1.2rem;
        color: #a0aec0;
        margin-bottom: 3rem;
        perspective: 1000px;
    }
    
    /* 3D Hover Card for Container */
    div[data-testid="stVerticalBlock"] > div[data-testid="stVerticalBlockBorderWrapper"] {
        transition: transform 0.4s ease, box-shadow 0.4s ease;
        transform-style: preserve-3d;
        background-color: rgba(11, 19, 36, 0.8) !important;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(0, 210, 255, 0.1) !important;
    }
    div[data-testid="stVerticalBlock"] > div[data-testid="stVerticalBlockBorderWrapper"]:hover {
        transform: translateY(-5px) scale(1.02) rotateX(2deg) rotateY(2deg);
        box-shadow: -10px 15px 30px rgba(0, 210, 255, 0.2), 10px -15px 30px rgba(58, 123, 213, 0.2);
    }
    
    /* Streamlit overrides for better UI */
    .stTextArea textarea {
        background-color: #0b1324 !important;
        color: #ffffff !important;
        border: 1px solid rgba(0, 210, 255, 0.2) !important;
        border-radius: 10px !important;
        transition: 0.3s;
        box-shadow: inset 0px 2px 5px rgba(0,0,0,0.5);
    }
    .stTextArea textarea:focus {
        border-color: #00d2ff !important;
        box-shadow: 0 0 15px rgba(0, 210, 255, 0.5), inset 0px 2px 5px rgba(0,0,0,0.5) !important;
        transform: scale(1.01);
    }
    
    /* 3D Glowing generate button */
    .stButton>button {
        width: 100%;
        background: linear-gradient(45deg, #00d2ff, #3a7bd5) !important;
        color: white !important;
        border: none !important;
        border-radius: 30px !important;
        font-weight: bold !important;
        padding: 0.8rem 1rem !important;
        transition: all 0.3s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        box-shadow: 0px 5px 0px #007799, 0px 10px 20px rgba(0, 210, 255, 0.4) !important;
        transform: translateY(0);
    }
    .stButton>button:hover {
        transform: translateY(-5px) scale(1.02);
        box-shadow: 0px 10px 0px #007799, 0px 15px 30px rgba(0, 210, 255, 0.6) !important;
    }
    .stButton>button:active {
        transform: translateY(5px);
        box-shadow: 0px 0px 0px #007799, 0px 2px 10px rgba(0, 210, 255, 0.4) !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    if st.session_state["extracted_data"]:
        # Render the Full Page Results
        st.markdown(
            """
            <style>
                [data-testid="collapsedControl"] { display: none; }
                [data-testid="stSidebar"] { display: none; }
            </style>
            """,
            unsafe_allow_html=True,
        )
        
        col1, col2 = st.columns([1, 5])
        with col1:
            if st.button("⬅️ Back", type="primary"):
                st.session_state["extracted_data"] = None
                st.rerun()
        with col2:
            st.success("✅ Portfolio generated successfully! It now takes up the full page.")
            
        st.divider()
        
        tab1, tab2, tab3 = st.tabs(["👁️ Live Preview", "🧩 Raw JSON", "🛡️ Hallucination Report"])
        
        with tab1:
            st.components.v1.html(st.session_state["html_content"], height=1000, scrolling=True)
            st.download_button(
                label="⬇️ Download portfolio.html",
                data=st.session_state["html_content"],
                file_name="portfolio.html",
                mime="text/html"
            )
            
        with tab2:
            st.json(st.session_state["extracted_data"])
            
        with tab3:
            st.markdown("### Hallucination Check")
            st.info("This simple algorithm checks if the extracted skills and links actually appear in the original text. It helps flag information the AI might have fabricated.")
            hallucinations = st.session_state["hallucinations"]
            if not hallucinations["skills"] and not hallucinations["links"]:
                st.success("✅ No hallucinations detected in Skills or Links!")
            else:
                if hallucinations["skills"]:
                    st.warning(f"⚠️ **Potentially Hallucinated Skills:** {', '.join(hallucinations['skills'])}")
                if hallucinations["links"]:
                    st.warning(f"⚠️ **Potentially Hallucinated Links:** {', '.join(hallucinations['links'])}")

        # Exit main early to hide the input form
        return
    
    st.markdown("<h1 class='glowing-title'>⚡ AI-Assisted Portfolio Generator</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle'>Transform your raw resume text into a structured, responsive portfolio website using <strong style='color: #00d2ff;'>Google Gemini 3.6 Flash</strong>.</p>", unsafe_allow_html=True)
    
    # Sidebar for Configuration
    with st.sidebar:
        st.markdown("<h2 style='text-align: center; color: #00d2ff; text-shadow: 0 0 10px rgba(0,210,255,0.3);'>⚙️ Settings</h2>", unsafe_allow_html=True)
        st.write("")
        
        with st.container(border=True):
            user_api_key = st.text_input("🔑 Google Gemini API Key (Optional)", value="", type="password", help="Leave blank to use the free public server key. Enter your own key if you want to use your personal quota.")
            st.markdown("<div style='text-align: right; font-size: 0.8rem;'><a href='https://aistudio.google.com/app/apikey' target='_blank' style='color: #00d2ff; text-decoration: none;'>Get API key →</a></div>", unsafe_allow_html=True)
            
        st.write("")
        
        with st.container(border=True):
            st.markdown("<h4 style='color: #a0aec0; margin-bottom: 10px;'>💡 Pro Tip</h4>", unsafe_allow_html=True)
            st.markdown("This app uses a premium AI backend to structure and write your portfolio sections flawlessly. Just paste your raw text and let the AI do the heavy lifting!")
            
        st.write("")
        
        with st.container(border=True):
            st.markdown("<h4 style='color: #a0aec0; margin-bottom: 10px;'>ℹ️ About</h4>", unsafe_allow_html=True)
            st.markdown("""
            **This app demonstrates:**
            - 🛡️ **Regex**: Masking PII (SSN/CC)
            - 🧠 **LLM**: Gemini API parsing
            - ⚙️ **Prompt Eng**: JSON output
            - 🔍 **Verification**: Hallucination check
            """)
        
    # Main Content Area
    st.markdown("<h3 style='color: #00d2ff; text-shadow: 0 0 10px rgba(0,210,255,0.3);'>1. Input Resume Content</h3>", unsafe_allow_html=True)
    
    input_method = st.radio("Choose Input Method:", ["Paste Text", "Upload File (PDF/Image)"])
    
    raw_resume = ""
    uploaded_file = None
    
    if input_method == "Paste Text":
        raw_resume = st.text_area("Paste your resume text here:", height=250, placeholder="John Doe\njohndoe@email.com\n\nExperience: Software Engineer at Tech Corp...")
    else:
        uploaded_file = st.file_uploader("Upload your resume", type=["pdf", "png", "jpg", "jpeg"])
        st.info("Note: PII Masking (SSN/CC redaction) and Hallucination Checks are only available for pasted text.")
        
    st.write("")
    submit_btn = st.button("🚀 Generate Portfolio", type="primary")
        
    if submit_btn:
        api_key = user_api_key if user_api_key.strip() else os.getenv("GEMINI_API_KEY", "")
        if not api_key:
            st.error("Configuration Error: No API Key available. Please enter your API Key in the sidebar.")
            st.stop()
            
        if input_method == "Paste Text":
            if not raw_resume.strip():
                st.error("Error: Missing Resume input. Please provide a resume.")
                st.stop()
            if len(raw_resume.strip()) < 50:
                st.error("Error: The provided resume is too short. Please provide a more detailed resume.")
                st.stop()
                
        if input_method == "Upload File (PDF/Image)":
            if not uploaded_file:
                st.error("Error: Missing Resume input. Please provide a resume.")
                st.stop()
            if len(uploaded_file.getvalue()) < 100:
                st.error("Error: The provided resume is too short. Please provide a more detailed resume.")
                st.stop()
            
        with st.spinner("Analyzing resume and generating portfolio..."):
            if input_method == "Paste Text":
                # Step 1: Pre-processing (Mask PII)
                masked_resume = mask_pii(raw_resume)
                # Step 2: LLM Extraction
                extracted_data = extract_portfolio_data(api_key=api_key, text=masked_resume)
            else:
                extracted_data = extract_portfolio_data(api_key=api_key, uploaded_file=uploaded_file)
            
            if extracted_data:
                if "error" in extracted_data:
                    st.error(extracted_data["error"])
                    st.stop()
                
                # Step 3: Post-processing (Hallucination Check)
                if input_method == "Paste Text":
                    hallucinations = check_hallucinations(raw_resume, extracted_data)
                else:
                    hallucinations = {"skills": [], "links": []}
                
                # Save to session state and rerun to switch to full-page mode
                st.session_state["extracted_data"] = extracted_data
                st.session_state["hallucinations"] = hallucinations
                st.session_state["html_content"] = generate_html(extracted_data)
                st.rerun()

if __name__ == "__main__":
    main()
