import asyncio
import os
import re
import json
import urllib.request
import pandas as pd
from dotenv import load_dotenv
from playwright.async_api import async_playwright
from langchain_groq import ChatGroq

# Force load environment variables
load_dotenv()

# =========================================================
# THE BASE RESUME (STRUCTURED FOR ATS TEMPLATE INJECTION)
# =========================================================
BASE_RESUME_DATA = {
    "name": "SANUGULA SHIVAKUMAR YADAV",
    "contact": {
        "linkedin": "www.linkedin.com/in/shivakumarsanugula",
        "email": "shivakumarsanugula97@gmail.com",
        "mobile": "+91-9700119139"
    },
    "education": [
        {
            "institution": "Christu Jyothi Institute of Technology & Science",
            "degree": "Bachelor of Technology in Electrical & Electronics Engineering; CGPA: 6.99",
            "period": "JUL 2017 - NOV 2020",
            "location": "Jangaon, Telangana, India"
        },
        {
            "institution": "Government Polytechnic",
            "degree": "Diploma in Electrical & Electronics Engineering; CGPA: 81.84%",
            "period": "AUG 2014 - APR 2017",
            "location": "Vikarabad, Telangana, India"
        },
        {
            "institution": "Holy Cross High School",
            "degree": "SSC; CGPA: 8.2",
            "period": "MAR 2014",
            "location": "Jangaon, Telangana, India"
        }
    ],
    "work_experience": [
        {
            "company": "Cospowers New Energy Pvt. Ltd",
            "role": "Graduate Engineer Trainee",
            "period": "NOV 2025 - Present",
            "location": "Hyderabad, Telangana, India",
            "bullets": [
                "Design, develop, and validate Battery Management System (BMS) hardware and embedded software for lithium-ion battery packs.",
                "Repaired and validated BMS circuit boards for 48V modules: component-level checks, replacements, and post-repair functional verification.",
                "Develop and implement SOC (State of Charge), SOH (State of Health), and SOP (State of Power) estimation algorithms.",
                "Lead module-level testing for 48V-100Ah and 48V-75Ah telecom battery modules, ensuring compliance with SOPs, internal QA standards, and manufacturing specifications.",
                "Supported R&D cell testing projects (for approvals such as BSNL-TSEC) across various C-rates; prepared detailed test logs and comparative analyses.",
                "Performed complete BMS validation testing, including voltage sensing, current measurement, protection logic, and communication checks for KELTRON manufactured systems.",
                "Evaluated Root Cause Analysis (RCA), performance imbalance, safety risks, and degradation behavior caused by integrating aged and new battery modules within the same battery system."
            ]
        },
        {
            "company": "Ernst & Young",
            "role": "Senior Data Analyst",
            "period": "DEC 2021 - AUG 2022",
            "location": "Hyderabad, Telangana, India",
            "bullets": [
                "Part of the Forensic team responsible for fraud detection, evidence gathering, data analysis, and reporting on short-term global projects.",
                "Conducted financial analysis for a major manufacturing client on P2P, Travel, and expenses; helped develop visualization dashboards.",
                "Developed ERP platforms with a unified data model under Digital Integrity Analytics (DIA); performed data mapping between source and target systems.",
                "Maintained versions of data models for production, testing, and development."
            ]
        },
        {
            "company": "Lotus Wave Software Solutions Pvt Ltd",
            "role": "DevOps Engineer & Data Analyst",
            "period": "JAN 2021 - DEC 2021",
            "location": "Hyderabad, Telangana, India",
            "bullets": [
                "Implemented DevOps automation, reducing deployment time-to-market by 40% and production defects by 50%.",
                "Developed tools to automate release activities and created single-click deployment jobs for environment upgrades.",
                "Established client-less deployment infrastructure, minimizing manual server interactions; deployment time cut by 90%."
            ]
        },
        {
            "company": "Besant Technologies Pvt Ltd",
            "role": "Data Science Intern",
            "period": "SEP 2020 - JAN 2021",
            "location": "Bangalore, Karnataka, India",
            "bullets": [
                "Conducted data preprocessing, feature engineering, and model evaluation for quality assurance.",
                "Managed master data, including creation, updates, and deletions.",
                "Created dashboards using Power BI and Tableau."
            ]
        }
    ],
    "certifications": [
        "Microsoft Certified: Power BI Data Analyst Associate",
        "Certified Associate in Python Programming",
        "Tableau Desktop & MS SQL Server Training",
        "Data Science and Machine Learning Training"
    ]
}

# =========================================================
# HELPER: API KEY ROTATION ENGINE
# =========================================================
def robust_llm_invoke(prompt_text: str, temperature: float = 0.0, model: str = "llama-3.3-70b-versatile") -> str:
    """Invokes Groq LLM and automatically rotates API keys if rate limited."""
    api_keys = [os.getenv("GROQ_API_KEY_1"), os.getenv("GROQ_API_KEY_2"), os.getenv("GROQ_API_KEY_3")]
    valid_keys = [key for key in api_keys if key and key.strip()]
    
    if not valid_keys:
        raise ValueError("[Error] No GROQ_API_KEYs found in environment/secrets!")

    for index, key in enumerate(valid_keys):
        try:
            llm = ChatGroq(model=model, temperature=temperature, groq_api_key=key, max_retries=0)
            return llm.invoke(prompt_text).content
        except Exception as e:
            error_message = str(e).lower()
            if "429" in error_message or "rate limit" in error_message:
                if index < len(valid_keys) - 1:
                    print(f"\n[System] Warning: API Key {index + 1} exhausted. Switching to Key {index + 2}...")
                    continue
                else:
                    print("\n[System] FATAL: All API keys exhausted!")
                    raise e
            else:
                raise e

# =========================================================
# HELPER: DATA CLEANING & DEDUPLICATION
# =========================================================
def clean_jobs_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty: return df
    text_cols = ['Source', 'Search Keyword', 'Location', 'Job Title', 'Company', 'Posted Date', 'Key Skills', 'Experience', 'Apply Link']
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).apply(lambda x: re.sub(r'\s+', ' ', x).strip())

    df = df.drop_duplicates(subset=['Apply Link'], keep='first')
    df = df.drop_duplicates(subset=['Job Title', 'Company'], keep='first')
    df = df[~df['Job Title'].str.lower().isin(['', 'unknown', 'none', 'nan', 'null'])]
    df = df[~df['Company'].str.lower().isin(['', 'unknown', 'none', 'nan', 'null'])]
    
    column_order = [
        'Source', 'Search Keyword', 'Location', 'Job Title', 'Company', 
        'Posted Date', 'Key Skills', 'Experience', 'Eligible_4_to_5_Years', 'Apply Link'
    ]
    df = df[[col for col in column_order if col in df.columns]]
    return df.reset_index(drop=True)

# =========================================================
# HELPER: AI JD PARSER 
# =========================================================
def extract_skills_and_experience(jd_text: str) -> dict:
    if not jd_text or len(jd_text) < 50:
        return {"Key Skills": "N/A", "Experience": "N/A", "Match": "No"}
    
    prompt = f"""
    You are a data extractor. Analyze the job description below.
    Extract the key skills and required experience.
    Determine if a candidate with 4 to 5 years of experience is a valid fit (Yes or No).
    
    You MUST respond with ONLY a raw JSON object:
    {{
        "skills": "comma-separated list of skills",
        "experience_summary": "1 sentence summarizing the required experience",
        "eligible_4_to_5_years": "Yes" or "No"
    }}

    Job Description:
    {jd_text[:2000]} 
    """
    try:
        response_text = robust_llm_invoke(prompt, temperature=0.0, model="llama-3.1-8b-instant")
        clean_json_str = response_text.replace("```json", "").replace("```", "").strip()
        data = json.loads(clean_json_str)
        return {
            "Key Skills": data.get("skills", "Not specified"), 
            "Experience": data.get("experience_summary", "Not specified"), 
            "Match": data.get("eligible_4_to_5_years", "No")
        }
    except Exception:
        return {"Key Skills": "Extraction Failed", "Experience": "Extraction Failed", "Match": "Error"}

# =========================================================
# 1A & 1B. SCRAPING AGENTS
# =========================================================
async def linkedin_agent(keyword_list, location_list, max_pages, context) -> list:
    page = await context.new_page()
    jobs = []
    for location in location_list:
        for keyword in keyword_list:
            for page_num in range(max_pages):
                start_param = page_num * 25
                linkedin_url = f"https://www.linkedin.com/jobs/search/?keywords={keyword}&location={location}&f_TPR=r86400&start={start_param}" 
                try:
                    await page.goto(linkedin_url, wait_until="domcontentloaded", timeout=15000)
                    await page.wait_for_timeout(2000)
                    for _ in range(3):
                        await page.keyboard.press("PageDown")
                        await page.wait_for_timeout(1000)
                        
                    job_cards = await page.locator(".base-search-card").all()
                    if not job_cards: break
                        
                    for card in job_cards:
                        try:
                            try: posted_date = await card.locator("time").first.inner_text()
                            except: posted_date = "Recent"
                            if re.search(r'([2-9]|\d{2,})\+?\s*days?|week|month', posted_date.lower()): continue
                            
                            title = (await card.locator(".base-search-card__title").inner_text()).strip()
                            company = (await card.locator(".base-search-card__subtitle").inner_text()).strip()
                            link = (await card.locator(".base-card__full-link").get_attribute("href")).strip()
                            
                            jobs.append({"Source": "LinkedIn", "Search Keyword": keyword, "Location": location, "Job Title": title, "Company": company, "Posted Date": posted_date, "Apply Link": link, "Key Skills": "Pending...", "Experience": "Pending...", "Eligible_4_to_5_Years": "Pending..."})
                        except: continue
                except: break
    await page.close()
    return jobs

async def naukri_agent(keyword_list, location_list, max_pages, context) -> list:
    page = await context.new_page()
    jobs = []
    for location in location_list:
        for keyword in keyword_list:
            formatted_keyword = keyword.replace(' ', '-')
            formatted_location = location.replace(' ', '-')
            for page_num in range(1, max_pages + 1):
                naukri_url = f"https://www.naukri.com/{formatted_keyword}-jobs-in-{formatted_location}{'-'+str(page_num) if page_num > 1 else ''}?jobAge=1&sort=r"
                try:
                    await page.goto(naukri_url, wait_until="domcontentloaded", timeout=15000)
                    await page.wait_for_timeout(2000)
                    for _ in range(3):
                        await page.keyboard.press("PageDown")
                        await page.wait_for_timeout(1000)
                    
                    naukri_cards = await page.locator(".srp-jobtuple-wrapper, .jobTuple, .cust-job-tuple").all()
                    if not naukri_cards: break
                    
                    for card in naukri_cards:
                        try:
                            try: posted_date = (await card.locator(".job-post-day, .type").first.inner_text()).strip()
                            except: posted_date = "Recent"
                            if re.search(r'([2-9]|\d{2,})\+?\s*days?|week|month', posted_date.lower()): continue
                            
                            title_elem = card.locator("a.title").first
                            title = (await title_elem.inner_text()).strip()
                            company = (await card.locator("a.comp-name, .companyInfo").first.inner_text()).strip()
                            link = (await title_elem.get_attribute("href")).strip()
                            
                            jobs.append({"Source": "Naukri", "Search Keyword": keyword, "Location": location, "Job Title": title, "Company": company, "Posted Date": posted_date, "Apply Link": link, "Key Skills": "Pending...", "Experience": "Pending...", "Eligible_4_to_5_Years": "Pending..."})
                        except: continue
                except: break
    await page.close()
    return jobs

# =========================================================
# 1. MAIN EXTRACTION ORCHESTRATOR
# =========================================================
async def scrape_jobs(keywords: str, locations: str, max_pages: int = 5) -> str:
    keyword_list = [k.strip() for k in keywords.split(',')]
    location_list = [l.strip() for l in locations.split(',')]
    
    print(f"\n[System] Phase 1: Scraping Jobs...")
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True, slow_mo=200)
        context = await browser.new_context(viewport={'width': 1920, 'height': 1080})
        
        linkedin_results, naukri_results = await asyncio.gather(
            linkedin_agent(keyword_list, location_list, max_pages, context),
            naukri_agent(keyword_list, location_list, max_pages, context)
        )
        
        # ZIPPER MERGE
        raw_jobs_data = [job for pair in zip(linkedin_results, naukri_results) for job in pair]
        min_len = min(len(linkedin_results), len(naukri_results))
        raw_jobs_data.extend(linkedin_results[min_len:])
        raw_jobs_data.extend(naukri_results[min_len:])
        
        if raw_jobs_data:
            df_temp = pd.DataFrame(raw_jobs_data)
            df_temp = df_temp.drop_duplicates(subset=['Apply Link'], keep='first')
            df_temp = df_temp.drop_duplicates(subset=['Job Title', 'Company'], keep='first')
            raw_jobs_data = df_temp.to_dict('records')

        max_ai_jobs = 40
        if len(raw_jobs_data) > max_ai_jobs:
            raw_jobs_data = raw_jobs_data[:max_ai_jobs]
                    
        print(f"\n[System] Phase 1.5: Reviewing {len(raw_jobs_data)} jobs via AI...")
        eval_page = await context.new_page()
        for index, job in enumerate(raw_jobs_data, 1):
            try:
                await eval_page.goto(job['Apply Link'], wait_until="domcontentloaded", timeout=15000)
                await eval_page.wait_for_timeout(2000)
                try:
                    if job['Source'] == 'LinkedIn': jd_text = await eval_page.locator(".description__text, .show-more-less-html__markup").first.inner_text()
                    else: jd_text = await eval_page.locator(".job-desc, .dang-inner-html, .styles_JBD__text__oBhw2").first.inner_text()
                except: jd_text = await eval_page.locator("body").inner_text()
                
                ai_extracted = extract_skills_and_experience(jd_text)
                job['Key Skills'] = ai_extracted['Key Skills']
                job['Experience'] = ai_extracted['Experience']
                job['Eligible_4_to_5_Years'] = ai_extracted['Match']
            except:
                job['Key Skills'] = "Error"
                job['Experience'] = "Error"
                job['Eligible_4_to_5_Years'] = "No Data"
                
        await browser.close()
        
    if raw_jobs_data:
        df_clean = clean_jobs_dataframe(pd.DataFrame(raw_jobs_data))
        df_clean.to_excel("cleaned_jobs_extract_2.xlsx", index=False, engine='openpyxl')
        return f"Success: Extracted {len(df_clean)} jobs."
    return "Failed: No jobs found."

# =========================================================
# EXACT ATS HTML/CSS TEMPLATE GENERATOR
# =========================================================
def build_exact_ats_html(tailored_data: dict) -> str:
    """Generates clean HTML matching ShivaKumar_Resume (2)_2.pdf layout."""
    
    # Extract tailored data
    summary = tailored_data.get("summary", "")
    skills = tailored_data.get("skills_summary", {})
    work_exp = tailored_data.get("work_experience", [])
    
    # Skills HTML construction
    skills_html = ""
    for category, val in skills.items():
        val_str = ", ".join(val) if isinstance(val, list) else str(val)
        skills_html += f"<li><strong>{category}:</strong> {val_str}</li>"

    # Education HTML construction
    edu_html = ""
    for edu in BASE_RESUME_DATA["education"]:
        edu_html += f"""
        <div class="entry-header">
            <div class="title-company">
                <strong>{edu['institution']}</strong><br/>
                <span>{edu['degree']}</span>
            </div>
            <div class="period-location">
                <span>{edu['period']}</span><br/>
                <span>{edu['location']}</span>
            </div>
        </div>
        """

    # Work Experience HTML construction
    work_html = ""
    for job in work_exp:
        bullets_html = "".join([f"<li>{b}</li>" for b in job.get("bullets", [])])
        work_html += f"""
        <div class="job-block">
            <div class="entry-header">
                <div class="title-company">
                    <strong>{job['company']}</strong><br/>
                    <em>{job['role']}</em>
                </div>
                <div class="period-location">
                    <span>{job['period']}</span><br/>
                    <span>{job['location']}</span>
                </div>
            </div>
            <ul>
                {bullets_html}
            </ul>
        </div>
        """

    # Certifications HTML construction
    cert_html = "".join([f"<li>{c}</li>" for c in BASE_RESUME_DATA["certifications"]])

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8"/>
        <style>
            @page {{
                size: A4;
                margin: 0.5in 0.6in;
            }}
            body {{
                font-family: 'Calibri', 'Arial', sans-serif;
                font-size: 10pt;
                line-height: 1.25;
                color: #111111;
                margin: 0;
                padding: 0;
            }}
            .header {{
                text-align: center;
                margin-bottom: 12px;
            }}
            .header h1 {{
                font-size: 16pt;
                margin: 0 0 4px 0;
                text-transform: uppercase;
                letter-spacing: 0.5px;
            }}
            .header .contact-info {{
                font-size: 9.5pt;
            }}
            .header .contact-info a {{
                color: #111111;
                text-decoration: none;
            }}
            .section-title {{
                font-size: 11pt;
                font-weight: bold;
                text-transform: uppercase;
                border-bottom: 1px solid #222222;
                margin-top: 10px;
                margin-bottom: 6px;
                padding-bottom: 1px;
                letter-spacing: 0.5px;
            }}
            .summary-text {{
                margin: 4px 0 8px 0;
                text-align: justify;
            }}
            .entry-header {{
                display: flex;
                justify-content: space-between;
                align-items: flex-start;
                margin-top: 6px;
                margin-bottom: 3px;
            }}
            .title-company {{
                text-align: left;
            }}
            .period-location {{
                text-align: right;
                white-space: nowrap;
                font-size: 9.5pt;
            }}
            ul {{
                margin: 3px 0 6px 0;
                padding-left: 18px;
            }}
            li {{
                margin-bottom: 2px;
                text-align: justify;
            }}
            .job-block {{
                margin-bottom: 6px;
            }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>{BASE_RESUME_DATA['name']}</h1>
            <div class="contact-info">
                LinkedIn: {BASE_RESUME_DATA['contact']['linkedin']} | 
                Email: {BASE_RESUME_DATA['contact']['email']} | 
                Mobile: {BASE_RESUME_DATA['contact']['mobile']}
            </div>
        </div>

        <div class="section-title">PROFESSIONAL SUMMARY</div>
        <div class="summary-text">{summary}</div>

        <div class="section-title">EDUCATION</div>
        {edu_html}

        <div class="section-title">SKILLS SUMMARY</div>
        <ul>
            {skills_html}
        </ul>

        <div class="section-title">WORK EXPERIENCE</div>
        {work_html}

        <div class="section-title">CERTIFICATIONS</div>
        <ul>
            {cert_html}
        </ul>
    </body>
    </html>
    """
    return html

# =========================================================
# PDF RENDERER (PLAYWRIGHT)
# =========================================================
async def render_pdf_from_html(html_content: str, output_pdf_path: str, context):
    """Renders exact HTML layout directly to ATS PDF."""
    temp_html_path = output_pdf_path.replace(".pdf", ".html")
    with open(temp_html_path, "w", encoding="utf-8") as f:
        f.write(html_content)
        
    page = await context.new_page()
    local_url = "file://" + urllib.request.pathname2url(os.path.abspath(temp_html_path))
    await page.goto(local_url)
    
    await page.pdf(
        path=output_pdf_path, 
        format="A4", 
        print_background=True,
        margin={"top": "0.4in", "right": "0.5in", "bottom": "0.4in", "left": "0.5in"}
    )
    
    await page.close()
    if os.path.exists(temp_html_path):
        os.remove(temp_html_path)

# =========================================================
# 2. BULK ATS RESUME GENERATOR
# =========================================================
async def generate_ats_resumes():
    """Generates tailored ATS-friendly PDF resumes using JSON structured tailoring."""
    try:
        df = pd.read_excel("cleaned_jobs_extract_2.xlsx")
        
        eligible_jobs = df[df['Eligible_4_to_5_Years'].astype(str).str.strip().str.lower() == 'yes']
        if eligible_jobs.empty:
            print("[System] Warning: No jobs matched 4-5 yrs fit strictly. Processing all extracted jobs instead...")
            eligible_jobs = df
            
        jobs_to_process = eligible_jobs.to_dict('records')
        
    except FileNotFoundError:
        return "Excel file not found. Run scraper first."

    os.makedirs("Tailored_Resumes", exist_ok=True)
    generated_files = []

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context()

        for idx, job in enumerate(jobs_to_process, 1):
            job_title = job.get('Job Title', 'Unknown Role')
            company = job.get('Company', 'Unknown Company')
            skills = job.get('Key Skills', '')
            
            clean_company = re.sub(r'[^a-zA-Z0-9]', '_', company)
            pdf_filename = f"Tailored_Resumes/ShivaKumar_Resume_{clean_company}.pdf"
            
            print(f"\n[System] Tailoring Resume {idx}/{len(jobs_to_process)} -> {job_title} at {company}...")
            
            prompt = f"""
            You are an expert technical ATS resume writer. Tailor the candidate's base resume for the following target job.
            
            TARGET ROLE: {job_title}
            TARGET COMPANY: {company}
            REQUIRED SKILLS: {skills}
            
            BASE RESUME DATA:
            {json.dumps(BASE_RESUME_DATA, indent=2)}
            
            Return ONLY a raw JSON object (no markdown formatting, no code blocks) matching this EXACT schema:
            {{
                "summary": "3-4 sentence professional summary tailored to {job_title} at {company}, highlighting relevant engineering/data/automation expertise.",
                "skills_summary": {{
                    "Languages": ["Python", "Verilog", "SQL"],
                    "Battery & Hardware": ["Battery physical & electrical inspection", "Data analysis", "LFP 48V Module manufacturing testing"],
                    "Testing & Quality": ["End-of-Line (EOL) testing", "SPC", "Performance validation"],
                    "Software & Tools": ["Power BI", "Tableau", "MS Excel", "ERP systems"],
                    "Interpersonal": ["Team management", "Problem solving", "Cross-functional communication"]
                }},
                "work_experience": [
                    {{
                        "company": "Cospowers New Energy Pvt. Ltd",
                        "role": "Graduate Engineer Trainee",
                        "period": "NOV 2025 - Present",
                        "location": "Hyderabad, Telangana, India",
                        "bullets": ["7 tailored bullet points emphasizing python, telemetry, hardware, data analysis, and testing..."]
                    }},
                    {{
                        "company": "Ernst & Young",
                        "role": "Senior Data Analyst",
                        "period": "DEC 2021 - AUG 2022",
                        "location": "Hyderabad, Telangana, India",
                        "bullets": ["4 tailored bullet points emphasizing financial analytics, data pipelines, SQL, Power BI..."]
                    }},
                    {{
                        "company": "Lotus Wave Software Solutions Pvt Ltd",
                        "role": "DevOps Engineer & Data Analyst",
                        "period": "JAN 2021 - DEC 2021",
                        "location": "Hyderabad, Telangana, India",
                        "bullets": ["3 tailored bullet points emphasizing deployment automation, CI/CD, Python..."]
                    }},
                    {{
                        "company": "Besant Technologies Pvt Ltd",
                        "role": "Data Science Intern",
                        "period": "SEP 2020 - JAN 2021",
                        "location": "Bangalore, Karnataka, India",
                        "bullets": ["3 tailored bullet points emphasizing preprocessing, dashboards..."]
                    }}
                ]
            }}
            """
            
            try:
                raw_json = robust_llm_invoke(prompt, temperature=0.2, model="llama-3.3-70b-versatile")
                clean_json_str = raw_json.replace("```json", "").replace("```", "").strip()
                tailored_data = json.loads(clean_json_str)
                
                # Build exact ATS HTML
                html_content = build_exact_ats_html(tailored_data)
                
                # Render to PDF
                await render_pdf_from_html(html_content, pdf_filename, context)
                    
                generated_files.append(pdf_filename)
                print(f" -> Successfully saved pixel-perfect PDF: {pdf_filename}")
                
            except Exception as e:
                print(f" -> Failed to generate resume for {company}: {e}")

        await browser.close()
        
    return f"Successfully generated {len(generated_files)} tailored PDF resumes in 'Tailored_Resumes' folder."


# =========================================================
# 4. AGENT EXECUTION
# =========================================================
def main():
    if not os.getenv("GROQ_API_KEY_1"):
        print("[Error] GROQ_API_KEY_1 is missing!")
        return

    print("==================================================")
    print(" STEP 1: PARALLEL WEB SCRAPING & AI JSON FILTERING")
    print("==================================================")
    
    search_status = asyncio.run(scrape_jobs("Data Engineer, Data Analyst", "Hyderabad, Bangalore", max_pages=3))
    print(f"\n[Step 1 Result] {search_status}\n")
    
    if "Success" in search_status:
        print("==================================================")
        print(" STEP 2: GENERATING PIXEL-PERFECT ATS PDF RESUMES")
        print("==================================================")
        
        tailor_status = asyncio.run(generate_ats_resumes())
        print(f"\n[Step 2 Result] {tailor_status}\n")
    else:
        print("[System] Stopping execution.")

if __name__ == "__main__":
    main()