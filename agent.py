import asyncio
import os
import re
import json
import pandas as pd
from dotenv import load_dotenv
from playwright.async_api import async_playwright
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent

# Force load environment variables from the .env file
load_dotenv()

# ---------------------------------------------------------
# HELPER: DATA CLEANING & DEDUPLICATION PIPELINE
# ---------------------------------------------------------
def clean_jobs_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Cleans, normalizes, deduplicates, and formats raw scraped job data."""
    if df.empty:
        return df

    text_cols = ['Source', 'Search Keyword', 'Location', 'Job Title', 'Company', 'Posted Date', 'Key Skills', 'Experience', 'Apply Link']
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).apply(lambda x: re.sub(r'\s+', ' ', x).strip())

    # Drop duplicates based on unique link AND combination of Title + Company
    df = df.drop_duplicates(subset=['Apply Link'], keep='first')
    df = df.drop_duplicates(subset=['Job Title', 'Company'], keep='first')

    # Filter out invalid/empty entries
    df = df[~df['Job Title'].str.lower().isin(['', 'unknown', 'none', 'nan', 'null'])]
    df = df[~df['Company'].str.lower().isin(['', 'unknown', 'none', 'nan', 'null'])]

    # NOTE: We are NO LONGER dropping jobs that fail the 2-3 years test. 
    # We will keep them in the Excel file so you can audit the AI's decisions.
    
    column_order = [
        'Source', 
        'Search Keyword', 
        'Location', 
        'Job Title', 
        'Company', 
        'Posted Date', 
        'Key Skills', 
        'Experience',
        'Eligible_2_to_3_Years', # Renamed for clarity
        'Apply Link'
    ]
    df = df[[col for col in column_order if col in df.columns]]
    return df.reset_index(drop=True)


# ---------------------------------------------------------
# HELPER: AI JD PARSER (UPDATED TO JSON)
# ---------------------------------------------------------
def extract_skills_and_experience(jd_text: str, llm: ChatGroq) -> dict:
    """Uses Groq to extract skills, experience, and evaluate via JSON."""
    if not jd_text or len(jd_text) < 50:
        return {"Key Skills": "N/A", "Experience": "N/A", "Match": "No"}
    
    # Force the model to output ONLY valid JSON
    prompt = f"""
    You are a data extractor. Analyze the job description below.
    Extract the key skills and required experience.
    Determine if a candidate with exactly 2 to 3 years of experience is a valid fit (True or False).
    
    You MUST respond with ONLY a raw JSON object in this exact format. Do not include markdown formatting or backticks:
    {{
        "skills": "comma-separated list of skills",
        "experience_summary": "1 sentence summarizing the required experience",
        "eligible_2_to_3_years": "Yes" or "No"
    }}

    Job Description:
    {jd_text[:2000]} 
    """
    
    try:
        response_text = llm.invoke(prompt).content
        
        # Clean up the string just in case the smaller model hallucinates markdown
        clean_json_str = response_text.replace("```json", "").replace("```", "").strip()
        
        data = json.loads(clean_json_str)
        
        return {
            "Key Skills": data.get("skills", "Not specified"), 
            "Experience": data.get("experience_summary", "Not specified"), 
            "Match": data.get("eligible_2_to_3_years", "No")
        }
    except json.JSONDecodeError:
        print("[Warning] AI Parsing Error: Model failed to return valid JSON.")
        return {"Key Skills": "Extraction Failed", "Experience": "Extraction Failed", "Match": "Parse Error"}
    except Exception as e:
        print(f"[Warning] AI Parsing Error: {e}")
        return {"Key Skills": "Extraction Failed", "Experience": "Extraction Failed", "Match": "Error"}


# ---------------------------------------------------------
# 1. PLAYWRIGHT EXTRACTION MODULE (Multi-Location, Multi-Page)
# ---------------------------------------------------------
async def scrape_jobs(keywords: str, locations: str, max_pages: int = 5) -> str:
    """Searches jobs on LinkedIn AND Naukri across multiple locations with strict 24-hour enforcement."""
    
    raw_jobs_data = []
    seen_links = set()
    seen_titles_companies = set()
    
    keyword_list = [k.strip() for k in keywords.split(',')]
    location_list = [l.strip() for l in locations.split(',')]
    
    api_key = os.getenv("GROQ_API_KEY")
    
    # Model configuration
    llm = ChatGroq(
        model="llama-3.1-8b-instant", 
        temperature=0, 
        groq_api_key=api_key,
        max_retries=5
    )
    
    print(f"\n[System] Phase 1: Publicly searching for {keyword_list} across {location_list} (Strictly < 24 Hours)...")
    
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False, slow_mo=200)
        context = await browser.new_context(
            viewport={'width': 1920, 'height': 1080},
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )
        page = await context.new_page()
        
        for location in location_list:
            for keyword in keyword_list:
                
                # --- 1A. LINKEDIN PAGINATION SCRAPING ---
                print(f"\n[System] Scraping LinkedIn for '{keyword}' in '{location}'...")
                for page_num in range(max_pages):
                    start_param = page_num * 25
                    linkedin_url = f"https://www.linkedin.com/jobs/search/?keywords={keyword}&location={location}&f_TPR=r86400&start={start_param}" 
                    
                    try:
                        await page.goto(linkedin_url, wait_until="domcontentloaded", timeout=15000)
                        await page.wait_for_timeout(3000)

                        for _ in range(4):
                            await page.keyboard.press("PageDown")
                            await page.wait_for_timeout(1000)
                            
                        job_cards = await page.locator(".base-search-card").all()
                        if not job_cards:
                            break
                            
                        page_new_jobs = 0
                        for card in job_cards:
                            try:
                                try:
                                    posted_date = await card.locator("time").first.inner_text()
                                except Exception:
                                    posted_date = "Recent"
                                
                                if re.search(r'([2-9]|\d{2,})\+?\s*days?|week|month', posted_date.lower()):
                                    continue
                                
                                title = (await card.locator(".base-search-card__title").inner_text()).strip()
                                company = (await card.locator(".base-search-card__subtitle").inner_text()).strip()
                                link = (await card.locator(".base-card__full-link").get_attribute("href")).strip()
                                
                                identifier = (title.lower(), company.lower())
                                if link in seen_links or identifier in seen_titles_companies:
                                    continue
                                    
                                seen_links.add(link)
                                seen_titles_companies.add(identifier)
                                
                                raw_jobs_data.append({
                                    "Source": "LinkedIn",
                                    "Search Keyword": keyword,
                                    "Location": location,
                                    "Job Title": title,
                                    "Company": company,
                                    "Posted Date": posted_date,
                                    "Apply Link": link,
                                    "Key Skills": "Pending...",
                                    "Experience": "Pending...",
                                    "Eligible_2_to_3_Years": "Pending..."
                                })
                                page_new_jobs += 1
                            except Exception:
                                continue
                                
                        print(f" -> LinkedIn [{location}] Page {page_num + 1}: Added {page_new_jobs} valid recent jobs.")
                        if page_new_jobs == 0:
                            break
                    except Exception as e:
                        print(f"[Warning] LinkedIn search stopped on page {page_num + 1}: {e}")
                        break

                # --- 1B. NAUKRI PAGINATION SCRAPING ---
                print(f"\n[System] Scraping Naukri for '{keyword}' in '{location}'...")
                formatted_keyword = keyword.replace(' ', '-')
                formatted_location = location.replace(' ', '-')
                
                for page_num in range(1, max_pages + 1):
                    if page_num == 1:
                        naukri_url = f"https://www.naukri.com/{formatted_keyword}-jobs-in-{formatted_location}?jobAge=1&sort=r"
                    else:
                        naukri_url = f"https://www.naukri.com/{formatted_keyword}-jobs-in-{formatted_location}-{page_num}?jobAge=1&sort=r"
                    
                    try:
                        await page.goto(naukri_url, wait_until="domcontentloaded", timeout=15000)
                        await page.wait_for_timeout(4000)
                        
                        for _ in range(4):
                            await page.keyboard.press("PageDown")
                            await page.wait_for_timeout(1000)
                        
                        naukri_cards = await page.locator(".srp-jobtuple-wrapper, .jobTuple, .cust-job-tuple").all()
                        if not naukri_cards:
                            break
                        
                        page_new_jobs = 0
                        for card in naukri_cards:
                            try:
                                try:
                                    posted_date = (await card.locator(".job-post-day, .type").first.inner_text()).strip()
                                except Exception:
                                    posted_date = "Recent"
                                
                                if re.search(r'([2-9]|\d{2,})\+?\s*days?|week|month', posted_date.lower()):
                                    continue
                                
                                title_elem = card.locator("a.title").first
                                title = (await title_elem.inner_text()).strip()
                                
                                company_elem = card.locator("a.comp-name, .companyInfo").first
                                company = (await company_elem.inner_text()).strip()
                                
                                link = (await title_elem.get_attribute("href")).strip()
                                
                                identifier = (title.lower(), company.lower())
                                if link in seen_links or identifier in seen_titles_companies:
                                    continue
                                    
                                seen_links.add(link)
                                seen_titles_companies.add(identifier)
                                    
                                raw_jobs_data.append({
                                    "Source": "Naukri",
                                    "Search Keyword": keyword,
                                    "Location": location,
                                    "Job Title": title,
                                    "Company": company,
                                    "Posted Date": posted_date,
                                    "Apply Link": link,
                                    "Key Skills": "Pending...",
                                    "Experience": "Pending...",
                                    "Eligible_2_to_3_Years": "Pending..."
                                })
                                page_new_jobs += 1
                            except Exception:
                                continue
                                
                        print(f" -> Naukri [{location}] Page {page_num}: Added {page_new_jobs} valid recent jobs.")
                        if page_new_jobs == 0:
                            break 
                    except Exception as e:
                        print(f"[Warning] Naukri search stopped on page {page_num}: {e}")
                        break

        # TOKEN OVERLOAD PROTECTION
        max_ai_jobs = 40
        if len(raw_jobs_data) > max_ai_jobs:
            print(f"\n[System] OVERLOAD PROTECTION: Capping AI analysis to top {max_ai_jobs} jobs.")
            raw_jobs_data = raw_jobs_data[:max_ai_jobs]
                    
        # LOOP 2: Visit Each Unique Job Link to Extract JD, Skills, and Experience
        print(f"\n[System] Phase 1.5: Reading {len(raw_jobs_data)} valid 24h job postings & mapping JSON...")
        for index, job in enumerate(raw_jobs_data, 1):
            print(f" -> [{index}/{len(raw_jobs_data)}] AI JSON Review: {job['Job Title']} at {job['Company']}")
            try:
                await page.goto(job['Apply Link'], wait_until="domcontentloaded", timeout=15000)
                await page.wait_for_timeout(2000)
                
                jd_text = ""
                try:
                    if job['Source'] == 'LinkedIn':
                        jd_locator = page.locator(".description__text, .show-more-less-html__markup").first
                    else: 
                        jd_locator = page.locator(".job-desc, .dang-inner-html, .styles_JBD__text__oBhw2").first
                    jd_text = await jd_locator.inner_text()
                except Exception:
                    jd_text = await page.locator("body").inner_text()
                
                ai_extracted = extract_skills_and_experience(jd_text, llm)
                job['Key Skills'] = ai_extracted['Key Skills']
                job['Experience'] = ai_extracted['Experience']
                job['Eligible_2_to_3_Years'] = ai_extracted['Match']
                
                await asyncio.sleep(2) 
                
            except Exception:
                job['Key Skills'] = "Could not load JD"
                job['Experience'] = "Could not load JD"
                job['Eligible_2_to_3_Years'] = "No Data"
                
        await browser.close()
        
    if raw_jobs_data:
        print("\n[System] Phase 1.6: Finalizing dataset (keeping all valid records for audit)...")
        df_raw = pd.DataFrame(raw_jobs_data)
        df_clean = clean_jobs_dataframe(df_raw)
        
        output_filename = "cleaned_jobs_extract_2.xlsx"
        df_clean.to_excel(output_filename, index=False, engine='openpyxl')
        
        return f"Success: Extracted {len(df_clean)} jobs to {output_filename}. Check the 'Eligible_2_to_3_Years' column to review the AI matches."
    
    return "Failed: No jobs found matching the base web scraping criteria."

# ---------------------------------------------------------
# 2. RESUME TAILORING MODULE (Analysis)
# ---------------------------------------------------------
async def tailor_resume_for_top_job() -> str:
    """Reads the cleaned extracted jobs and tailors a resume for the top MATCHING job."""
    try:
        df = pd.read_excel("cleaned_jobs_extract_2.xlsx")
        if df.empty:
            return "No valid jobs found to analyze."
            
        # We now filter dynamically during Phase 2 instead of deleting the data in Phase 1
        eligible_jobs = df[df['Eligible_2_to_3_Years'].astype(str).str.lower().str.contains('yes', na=False)]
        
        if eligible_jobs.empty:
            return "Audit Complete: No jobs in the spreadsheet were flagged as a 'Yes' for 2-3 years experience by the AI."
            
        top_job = eligible_jobs.iloc[0].to_dict()
        
    except FileNotFoundError:
        return "Excel file not found. Please run the job search tool first."

    source = top_job.get('Source', 'Unknown Portal')
    job_title = top_job.get('Job Title', 'Unknown Title')
    company = top_job.get('Company', 'Unknown Company')
    location = top_job.get('Location', 'Unknown Location')
    skills = top_job.get('Key Skills', '')
    experience = top_job.get('Experience', '')

    print(f"\n[System] Phase 2: Tailoring Resume for '{job_title}' at {company} in {location}...")
    
    my_background = """
    Expertise in building data automation tools and data pipelines to append multiple datasets into unified files for analysis. 
    Proficient in Python-based systems to process Excel and CSV workflows using pandas, openpyxl, and matplotlib. 
    Experience integrating telemetry data and performing complex battery degradation analysis, including cycle counting and state-of-health monitoring.
    """
    
    api_key = os.getenv("GROQ_API_KEY")
    
    llm = ChatGroq(
        model="llama-3.1-8b-instant", 
        temperature=0.7,
        groq_api_key=api_key,
        max_retries=5
    )
    
    prompt = f"""
    You are an expert technical recruiter. I am applying for the role of {job_title} at {company} in {location}.
    
    The job requires these skills: {skills}
    And this experience: {experience}
    
    Here is my core background:
    {my_background}
    
    Generate a highly tailored 3-sentence 'Professional Summary' for my resume that perfectly aligns my background with this specific job description. 
    Focus on data automation, pipelines, and Python.
    """
    
    response = llm.invoke(prompt)
    return f"Target Job: {job_title} at {company} ({location}) via {source}\nRequired Skills: {skills}\nRequired Experience: {experience}\n\nTailored Resume Summary:\n{response.content}"

# ---------------------------------------------------------
# 3. LANGGRAPH TOOL DEFINITIONS
# ---------------------------------------------------------
@tool
def search_jobs_tool(keywords: str, locations: str, max_pages: int = 5) -> str:
    """
    Use this tool to publicly search for job postings on BOTH LinkedIn and Naukri across multiple locations.
    """
    return asyncio.run(scrape_jobs(keywords, locations, max_pages=max_pages))

@tool
def analyze_and_tailor_tool() -> str:
    """Use this tool AFTER searching for jobs to read the Excel file and tailor a resume."""
    return asyncio.run(tailor_resume_for_top_job())

# ---------------------------------------------------------
# 4. AGENT EXECUTION
# ---------------------------------------------------------
def main():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("[Error] GROQ_API_KEY is not set in your .env file!")
        return

    print("[System] Initializing JSON-Powered Experience Filtering Agent...")
    
    llm = ChatGroq(
        model="llama-3.1-8b-instant", 
        temperature=0,
        groq_api_key=api_key,
        max_retries=5
    )
    
    agent_executor = create_react_agent(llm, tools=[search_jobs_tool, analyze_and_tailor_tool])
    
    user_prompt = """
    Step 1: Find 'Data Engineer' and 'Data Analyst' roles in Hyderabad and Bangalore posted in the last 24 hours. 
    Search BOTH LinkedIn and Naukri, crawling up to 5 pages per portal, per keyword, per location.
    Step 2: Once extracted and filtered, analyze the top matching job from the list and tailor my resume for it.
    """
    print(f"[User] {user_prompt}\n")
    
    response = agent_executor.invoke(
        {"messages": [("user", user_prompt)]}
    )
    
    print("\n==============================")
    print("[Final Agent Output]")
    print(response["messages"][-1].content)
    print("==============================\n")

if __name__ == "__main__":
    main()