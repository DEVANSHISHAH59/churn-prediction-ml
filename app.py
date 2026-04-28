import streamlit as st
import pandas as pd
from datetime import date, datetime
import json, os, random

# ── SILICON VALLEY & GLOBAL TECH NEWS ─────────────────────────────────────────
SV_NEWS = [
    {"title":"Anthropic hits $30B revenue in April 2026 — in early IPO talks with Goldman Sachs","url":"https://venturebeat.com/category/ai","source":"VentureBeat","date":"Apr 25, 2026","tag":"AI","relevance":"Anthropic Dublin is hiring - major growth signal"},
    {"title":"Google invests up to $40B in Anthropic at ~$350B valuation","url":"https://techcrunch.com/category/artificial-intelligence/","source":"TechCrunch","date":"Apr 25, 2026","tag":"AI","relevance":"Google and Anthropic Dublin partnerships expanding"},
    {"title":"Meta confirms 8,000 layoffs but doubles AI capex to $115-135B in 2026","url":"https://techcrunch.com/category/artificial-intelligence/","source":"TechCrunch","date":"Apr 25, 2026","tag":"AI","relevance":"Meta Dublin T&S team unaffected - capex shows AI commitment"},
    {"title":"DeepSeek releases V4-Pro open-source model - 1.6 trillion parameters","url":"https://venturebeat.com/category/ai","source":"VentureBeat","date":"Apr 24, 2026","tag":"AI","relevance":"Open-source AI boom = more LLM evaluation roles"},
    {"title":"EU AI Act full enforcement starts August 2, 2026 - 99 days away","url":"https://techcrunch.com/category/artificial-intelligence/","source":"TechCrunch","date":"Apr 25, 2026","tag":"Policy","relevance":"Your EU AI Act knowledge is urgently in demand right now"},
    {"title":"Microsoft launches 3 new foundational AI models: MAI-Transcribe, MAI-Voice, MAI-Image","url":"https://techcrunch.com/2026/04/02/microsoft-takes-on-ai-rivals-with-three-new-foundational-models/","source":"TechCrunch","date":"Apr 3, 2026","tag":"AI","relevance":"Microsoft Dublin expanding AI teams"},
    {"title":"OpenAI proposes public wealth funds and robot taxes as AI reshapes jobs","url":"https://techcrunch.com/2026/04/06/openais-vision-for-the-ai-economy-public-wealth-funds-robot-taxes-and-a-four-day-work-week/","source":"TechCrunch","date":"Apr 6, 2026","tag":"Policy","relevance":"T&S and AI governance roles growing in strategic importance"},
    {"title":"Silicon Valley confronts AI job panic - companies cite AI in announcing cuts","url":"https://techxplore.com/news/2026-04-hiring-humans-silicon-valley-ai.html","source":"TechXplore","date":"Apr 12, 2026","tag":"Jobs","relevance":"T&S and policy roles are protected - AI needs human oversight"},
    {"title":"Google Thinking Machines Lab multibillion deal - GB300 chip access","url":"https://techcrunch.com/2026/04/22/exclusive-google-deepens-thinking-machines-lab-ties-with-new-multi-billion-dollar-deal/","source":"TechCrunch","date":"Apr 22, 2026","tag":"AI","relevance":"Google AI expanding - Dublin office benefits"},
    {"title":"AI demand for ML engineers nearly doubled since 2023 - ManpowerGroup","url":"https://leaddev.com/hiring/how-ai-boom-bringing-silicon-valley-back-brink","source":"LeadDev","date":"Recent","tag":"Jobs","relevance":"Your LLM evaluation skills are at peak demand globally"},
    {"title":"AI reasoning with smaller models yields stronger performance at lower cost","url":"https://venturebeat.com/category/ai","source":"VentureBeat","date":"Apr 2026","tag":"AI","relevance":"LLM evaluation expertise more critical than ever"},
    {"title":"Four AI research trends for 2026 - continual learning, world models, agents","url":"https://venturebeat.com/technology/four-ai-research-trends-enterprise-teams-should-watch-in-2026","source":"VentureBeat","date":"2026","tag":"AI","relevance":"Stay informed for T&S and AI Analyst interviews"},
]

# ── ALL DUBLIN COMPANIES (comprehensive) ──────────────────────────────────────
ALL_COMPANIES = [
    # Big Tech - FAANG+
    ("Google / YouTube",    "G",   "Big Tech",     "~5,500 staff. European HQ Silicon Docks.",         "https://careers.google.com/jobs/results/?location=Dublin&q=trust+safety+analyst"),
    ("Meta",                "M",   "Big Tech",     "European HQ. T&S, Policy, Engineering.",           "https://www.metacareers.com/jobs?offices=Dublin&q=trust+integrity+analyst"),
    ("Microsoft",           "MS",  "Big Tech",     "~3,500 staff. Dublin since 1985.",                 "https://jobs.microsoft.com/en/search?q=analyst&lc=Dublin"),
    ("Amazon / AWS",        "AZ",  "Big Tech",     "Major Dublin presence. Cloud & ops roles.",        "https://www.amazon.jobs/en/search?location[]=IRL-Dublin"),
    ("Apple",               "AP",  "Big Tech",     "European HQ Cork, large Dublin office.",           "https://jobs.apple.com/en-us/search?location=dublin-DUB"),
    ("TikTok / ByteDance",  "T",   "Big Tech",     "Rapidly expanding T&S team. Hot!",                 "https://careers.tiktok.com/position?location=CT_211&query=trust+safety"),
    ("IBM",                 "IBM", "Big Tech",     "First American tech co in Ireland (1956).",        "https://www.ibm.com/employment/ie/"),
    ("Intel",               "INT", "Big Tech",     "Major campus. Engineering & data.",               "https://jobs.intel.com/en/search#q=dublin&t=Jobs"),
    ("Oracle",              "ORC", "Big Tech",     "European HQ Dublin. Large tech presence.",        "https://www.oracle.com/ie/corporate/careers/"),
    # AI
    ("OpenAI",              "OA",  "AI",           "Actively hiring T&S Dublin. TOP PRIORITY.",       "https://openai.com/careers"),
    ("Anthropic",           "AN",  "AI",           "Rapidly growing. $30B revenue April 2026.",       "https://www.anthropic.com/careers"),
    # Fintech & Payments
    ("Stripe",              "S",   "Fintech",      "European HQ Dublin. Risk & data roles.",          "https://stripe.com/jobs/search?l=Dublin&q=analyst"),
    ("Revolut",             "R",   "Fintech",      "Growing Dublin team. Risk, compliance.",          "https://www.revolut.com/careers/?location=Dublin"),
    ("PayPal",              "PP",  "Fintech",      "European HQ Dublin. Policy & ops.",               "https://careers.pypl.com/home/"),
    ("Mastercard",          "MC",  "Fintech",      "Dublin office. Analyst & risk roles.",            "https://careers.mastercard.com/us/en/search-results?keywords=analyst&location=Dublin"),
    ("Visa",                "V",   "Fintech",      "Dublin office. Data & policy roles.",             "https://corporate.visa.com/en/jobs/search?q=analyst&location=Dublin"),
    ("TrueLayer",           "TL",  "Fintech",      "Open banking. $270M raised. Stripe-backed.",      "https://truelayer.com/jobs/"),
    ("Monzo",               "MZ",  "Fintech",      "Digital bank. Dublin office growing.",            "https://monzo.com/careers/"),
    ("Fenergo",             "FE",  "Fintech",      "AI KYC platform. Irish HQ. Deloitte partner.",   "https://www.fenergo.com/company/careers/"),
    ("Block (Square)",      "BL",  "Fintech",      "Dublin office. Finance & ops roles.",             "https://careers.block.xyz/"),
    # SaaS & Cloud
    ("Salesforce",          "SF",  "SaaS",         "2,800 staff Dublin. Agentforce AI platform.",     "https://careers.salesforce.com/en/jobs/?search=analyst&location=Dublin"),
    ("HubSpot",             "HS",  "SaaS",         "European HQ Dublin. Analyst & PM roles.",        "https://www.hubspot.com/careers/jobs?q=analyst&countryCodes=IE"),
    ("Adobe",               "AD",  "SaaS",         "~4,000 staff Dublin Citywest. Expanding.",        "https://careers.adobe.com/us/en/search-results?keywords=analyst&location=Dublin"),
    ("Workday",             "WD",  "SaaS",         "Dublin office. HR tech & analyst roles.",        "https://www.workday.com/en-us/company/careers/open-positions.html"),
    ("Zendesk",             "ZD",  "SaaS",         "Dublin office. Support & analyst roles.",        "https://jobs.zendesk.com/us/en/search-results?keywords=analyst&location=Dublin"),
    ("Intercom",            "IC",  "SaaS",         "Founded in Dublin. Data & ops roles.",           "https://www.intercom.com/careers"),
    ("MongoDB",             "MDB", "SaaS",         "Dublin office. Engineering & ops.",              "https://www.mongodb.com/company/careers"),
    ("Dropbox",             "DB",  "SaaS",         "Dublin office. Operations roles.",               "https://jobs.dropbox.com/"),
    ("Squarespace",         "SQ",  "SaaS",         "Major Dublin hub. Operations & analyst.",        "https://www.squarespace.com/about/careers"),
    ("Klaviyo",             "KL",  "SaaS",         "Expanding Dublin team. Analyst roles.",          "https://www.klaviyo.com/careers"),
    ("Contentful",          "CF2", "SaaS",         "Dublin office. Operations & analyst.",           "https://www.contentful.com/careers/"),
    ("ServiceNow",          "SN",  "SaaS",         "Dublin office. Enterprise & analyst.",           "https://careers.servicenow.com/en/jobs/?search=analyst&location=Dublin"),
    ("Udemy",               "UD",  "SaaS",         "Dublin office. 17,000+ org customers.",          "https://about.udemy.com/careers/"),
    ("Workhuman",           "WH",  "SaaS",         "Co-HQ Dublin & Boston. HR analytics.",           "https://www.workhuman.com/careers/"),
    # Cybersecurity
    ("Cloudflare",          "CLF", "Security",     "Dublin office. Policy & analytics roles.",       "https://www.cloudflare.com/careers/jobs/?location=Dublin"),
    ("Tines",               "TN",  "Security",     "Irish startup. No-code security platform.",      "https://www.tines.com/careers"),
    ("Rapid7",              "R7",  "Security",     "Dublin office. Security analyst roles.",         "https://www.rapid7.com/careers/"),
    # Platforms & Marketplaces
    ("LinkedIn",            "LI",  "Platforms",    "European HQ Dublin. Analyst & PM roles.",       "https://careers.linkedin.com/jobs/search?keywords=analyst&location=Dublin"),
    ("Indeed",              "IN",  "Platforms",    "European HQ Dublin. Ops & product.",             "https://careers.indeed.com/"),
    ("Airbnb",              "AB",  "Platforms",    "Dublin office. Trust & policy roles.",           "https://careers.airbnb.com/positions/"),
    ("eBay",                "EB",  "Platforms",    "European HQ Dublin. Policy & ops.",              "https://careers.ebayinc.com/career-search/"),
    ("Etsy",                "ET",  "Platforms",    "Dublin office. Operations roles.",               "https://careers.etsy.com/"),
    ("Whatnot",             "WN",  "Platforms",    "T&S Senior Manager role open now!",              "https://www.whatnot.com/careers"),
    ("Pinterest",           "PI",  "Platforms",    "Dublin office. Product & data.",                 "https://www.pinterestcareers.com/"),
    ("Booking.com",         "BK",  "Platforms",    "Dublin office. Data & ops roles.",               "https://careers.booking.com/"),
    # Consulting
    ("Accenture",           "AC",  "Consulting",   "Massive Dublin presence. Major T&S contractor.", "https://www.accenture.com/ie-en/careers/jobsearch?jk=analyst&cl=Dublin"),
    ("Deloitte",            "DL",  "Consulting",   "Dublin HQ. BA, analytics, consulting.",         "https://apply.deloitte.com/careers/SearchJobs/analyst?3_56_3=5440"),
    ("EY",                  "EY",  "Consulting",   "Dublin HQ. FS consulting, BA roles.",           "https://www.ey.com/en_ie/careers"),
    ("PwC",                 "PW",  "Consulting",   "Dublin HQ. Risk, data, advisory.",              "https://www.pwc.ie/careers.html"),
    ("KPMG",                "KP",  "Consulting",   "Dublin HQ. Audit, analytics, advisory.",        "https://home.kpmg/ie/en/home/careers.html"),
    ("Capgemini",           "CG",  "Consulting",   "Dublin office. Tech consulting & BA.",          "https://www.capgemini.com/ie-en/careers/"),
    ("Cognizant",           "CO2", "Consulting",   "Dublin office. IT & business analyst.",         "https://careers.cognizant.com/global/en/search-results?keywords=analyst&location=Dublin"),
    ("Infosys",             "IF",  "Consulting",   "Dublin office. IT & BA roles.",                 "https://www.infosys.com/careers/apply.html"),
    ("Wipro",               "WI",  "Consulting",   "Dublin office. Tech & analyst roles.",          "https://careers.wipro.com/"),
    # Finance & Banking
    ("Citi",                "CI",  "Finance",      "Major Dublin office. Risk & ops analyst.",      "https://jobs.citi.com/search-jobs/Dublin"),
    ("JP Morgan",           "JP",  "Finance",      "Dublin office. Finance & data analyst.",        "https://careers.jpmorgan.com/global/en/search-jobs?location=Dublin"),
    ("Bank of America",     "BA2", "Finance",      "Dublin office. Analytics & risk.",              "https://careers.bankofamerica.com/en-us/search-jobs/Dublin"),
    ("AIB",                 "AIB", "Finance",      "Irish bank HQ. Data & BA roles.",               "https://aib.ie/careers"),
    ("Bank of Ireland",     "BOI", "Finance",      "Irish bank HQ. PO & BA roles. CSPO+.",          "https://careers.bankofireland.com"),
    ("Irish Life",          "IL",  "Finance",      "Dublin HQ. AI Governance Analyst open.",        "https://www.irishlife.ie/careers"),
    ("Davy",                "DV",  "Finance",      "Dublin HQ. PMO & analyst roles.",               "https://www.davy.ie/careers"),
    ("State Street",        "SS",  "Finance",      "Dublin office. Finance & data analyst.",        "https://careers.statestreet.com/"),
    ("Northern Trust",      "NT",  "Finance",      "Dublin office. Data & analytics.",              "https://www.northerntrust.com/united-states/what-we-do/careers"),
    ("Fidelity",            "FI",  "Finance",      "Dublin office. Tech & analyst roles.",          "https://jobs.fidelity.com/"),
    # Gaming & Media
    ("Activision Blizzard", "AG",  "Gaming",       "Dublin office. Trust & Safety roles.",          "https://careers.activisionblizzard.com/"),
    ("Riot Games",          "RG",  "Gaming",       "Dublin office. Policy & ops roles.",            "https://www.riotgames.com/en/work-with-us"),
    ("Electronic Arts",     "EA",  "Gaming",       "Dublin office. Data & product roles.",          "https://jobs.ea.com/"),
    # Startups & Scaleups
    ("LILT",                "LT",  "Startup",      "AI translation. Sequoia-backed. PO roles.",     "https://lilt.com/careers"),
    ("beqom",               "BQ",  "Startup",      "Pay analytics. PO role open. Remote-friendly.", "https://www.beqom.com/careers"),
    ("NextRoll",            "NR",  "Startup",      "AdRoll/MarTech. AI advertising platform.",     "https://www.nextroll.com/careers"),
    ("Grafana Labs",        "GL",  "Startup",      "Observability. Analytics & ops roles.",         "https://grafana.com/about/careers/"),
    ("Toast",               "TO",  "Startup",      "Restaurant tech. Dublin hybrid.",               "https://careers.toasttab.com/"),
    ("Ocuco",               "OC",  "Startup",      "Irish eyecare software. Dublin 15. BA/PO.",     "https://www.ocuco.com/company/careers/"),
    ("Manna",               "MA",  "Startup",      "Irish drone delivery. Growing team.",           "https://www.manna.aero/careers"),
    ("Workhuman",           "WH2", "Startup",      "HR tech. Co-HQ Dublin. Analyst roles.",        "https://www.workhuman.com/careers/"),
    ("Sojern",              "SJ",  "Startup",      "Travel AI. Dublin office. Data roles.",         "https://www.sojern.com/about/careers/"),
    ("CarGurus",            "CG2", "Startup",      "Auto marketplace. ~50 staff Dublin sales.",     "https://www.cargurus.com/about/careers"),
    ("Geneva Trading",      "GT",  "Startup",      "Prop trading tech. Dublin office.",             "https://www.genevatrading.com/careers/"),
]

st.set_page_config(page_title="Lets Get Hired - Devanshi", page_icon=":rocket:", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@400;500;600;700&family=DM+Sans:wght@400;500&display=swap');
html,body,[class*="css"]{font-family:'DM Sans',sans-serif;}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#1a0533,#0f0a1e)!important;}
[data-testid="stSidebar"] *{color:#c4b5fd!important;}
[data-testid="metric-container"]{background:#1a0533;border:1px solid #4c1d95;border-radius:14px;padding:1rem!important;}
[data-testid="stMetricValue"]{color:#a78bfa!important;font-size:2rem!important;}
.stButton>button{background:#7c3aed!important;color:white!important;border:none!important;border-radius:10px!important;}
.stButton>button:hover{background:#6d28d9!important;}
.stTabs [data-baseweb="tab-list"]{background:#1a0533;border-radius:12px;padding:4px;}
.stTabs [aria-selected="true"]{background:#7c3aed!important;color:white!important;}
.card{background:#1a0533;border:1px solid #2d1063;border-radius:14px;padding:1rem 1.2rem;margin-bottom:8px;}
.hero{background:linear-gradient(135deg,#4c1d95,#7c3aed);border-radius:18px;padding:1.5rem 2rem;color:white;margin-bottom:1.5rem;}
.quote-box{background:linear-gradient(135deg,#2d1063,#4c1d95);border-radius:16px;padding:1.2rem 1.5rem;margin-bottom:1.2rem;border-left:4px solid #a78bfa;}
.chip{padding:2px 10px;border-radius:20px;font-size:11px;font-weight:500;display:inline-block;}
.c-ts{background:#4c1d95;color:#ddd6fe;}
.c-ai{background:#1e3a5f;color:#bae6fd;}
.c-da{background:#064e3b;color:#a7f3d0;}
.c-po{background:#78350f;color:#fde68a;}
.c-ba{background:#7f1d1d;color:#fecaca;}
.c-sv{background:#1f2937;color:#9ca3af;}
.c-ap{background:#1e3a5f;color:#93c5fd;}
.c-in{background:#78350f;color:#fde68a;}
.c-of{background:#064e3b;color:#6ee7b7;}
.c-re{background:#7f1d1d;color:#fca5a5;}
.c-gh{background:#1f2937;color:#6b7280;}
.hot{background:#064e3b;color:#6ee7b7;padding:1px 7px;border-radius:8px;font-size:10px;font-weight:600;margin-left:5px;}
.newb{background:#1e3a5f;color:#93c5fd;padding:1px 7px;border-radius:8px;font-size:10px;font-weight:600;margin-left:5px;}
.sal{background:#2d1063;color:#a78bfa;padding:1px 7px;border-radius:8px;font-size:11px;margin-left:5px;border:1px solid #4c1d95;}
.priority-badge{background:#7c3aed;color:white;padding:1px 7px;border-radius:8px;font-size:10px;font-weight:600;margin-left:5px;}
.agency-card{background:#1a0533;border:1px solid #2d1063;border-radius:12px;padding:0.9rem 1rem;margin-bottom:8px;}
.tip-box{background:#1e3a5f;border:1px solid #1d4ed8;border-radius:12px;padding:1rem 1.2rem;margin-bottom:10px;}
.fun-fact{background:#064e3b;border:1px solid #065f46;border-radius:12px;padding:0.8rem 1rem;margin-bottom:8px;}
hr{border:none;border-top:1px solid #2d1063;margin:0.8rem 0;}
#MainMenu,footer,header{visibility:hidden;}
[data-testid="stSidebar"]{display:none!important;}
/* Top nav bar */
.topnav{background:linear-gradient(90deg,#1a0533,#2d1063);border-bottom:1px solid #4c1d95;padding:10px 20px;display:flex;align-items:center;gap:8px;flex-wrap:wrap;position:sticky;top:0;z-index:999;}
.topnav-title{font-family:Sora,sans-serif;font-size:15px;font-weight:600;color:#e8d5ff;margin-right:8px;white-space:nowrap;}
.topnav-btn{padding:5px 12px;border-radius:20px;font-size:12px;cursor:pointer;border:1px solid #2d1063;background:transparent;color:#c4b5fd;font-family:"DM Sans",sans-serif;white-space:nowrap;transition:all 0.15s;}
.topnav-btn:hover{background:#2d1063;color:#e8d5ff;}
.topnav-btn.active{background:#7c3aed;color:white;border-color:#7c3aed;}
</style>

""", unsafe_allow_html=True)

DATA_FILE = "/tmp/jobs_data.json"

QUOTES = [
    ("Devanshi, I believe in you. Your experience in LLM evaluation and Trust & Safety is rare and genuinely valuable. The right company is lucky to get you.", "Claude"),
    ("Every application is one step closer. You have exactly what Dublin's AI companies need right now.", "Claude"),
    ("You have survived 100% of your hardest days so far. This next chapter is yours.", "Claude"),
    ("Your combination of T&S operations, LLM evaluation and data skills is genuinely hard to find. Own it.", "Claude"),
    ("Devanshi, the redundancy is not a reflection of your worth. It is a door opening to something better and permanent.", "Claude"),
    ("You built dashboards, owned markets, evaluated LLMs and shaped policy. That is a senior profile. Apply senior.", "Claude"),
    ("Trust & Safety + AI + Dublin = you are in exactly the right place at exactly the right time.", "Claude"),
    ("Every expert was once a beginner who refused to give up. Keep going, Devanshi.", "Claude"),
    ("Your MSc, your CSPO, your 4-market experience - these are not small things. They are your edge.", "Claude"),
    ("The best time to plant a tree was 20 years ago. The second best time is now. Send the application.", "Claude"),
    ("Devanshi, OpenAI Dublin needs someone exactly like you. Believe that and apply today.", "Claude"),
    ("Difficult roads often lead to beautiful destinations. Your permanent role is ahead.", "Claude"),
    ("You are not starting over. You are starting from experience. Big difference.", "Claude"),
    ("One email today could change everything. Send it.", "Claude"),
    ("Your story is not over. The best chapter is still being written.", "Claude"),
    ("3 years of T&S at Meta is worth more than most people realise. Make sure they know that.", "Claude"),
]

FUN_FACTS = [
    "Dublin has 16 of the top 20 global tech companies - you are in the right city!",
    "There are 160,000+ IT professionals in Ireland - and T&S talent is the rarest of them all.",
    "OpenAI, TikTok and Meta are ALL actively hiring T&S in Dublin right now. Apply to all three.",
    "The average time to hire in Dublin tech is 4-6 weeks. Keep the pipeline full!",
    "Trust & Safety professionals with LLM experience are the most in-demand in EU AI compliance right now.",
    "Silicon Republic reports T&S roles in Ireland grew 40% in 2025. You are in a hot market.",
    "A CSPO cert + MSc Business Analytics puts you in the top 15% of candidates for Product Owner roles.",
    "Most Dublin tech jobs are filled within 3 applications to the right recruiter. Register with CPL and Morgan McKinley today.",
    "EU AI Act enforcement started 2026. Companies are urgently hiring AI governance and T&S analysts.",
    "80% of Dublin tech jobs are filled before they are even advertised - recruiters are your secret weapon.",
]

JOB_SEARCH_TIPS = [
    "Apply on Monday or Tuesday morning - hiring managers review applications at the start of the week.",
    "Personalise the first 2 lines of every cover letter to the specific company. It takes 3 minutes and doubles your callback rate.",
    "Follow up by email 5 days after applying. Most candidates never do this. It sets you apart.",
    "Connect with the hiring manager on LinkedIn BEFORE applying. It puts your name in their head.",
    "Register with at least 3 recruitment agencies. 80% of Dublin tech roles are filled through agencies before they are advertised.",
    "Set up job alerts on LinkedIn AND Indeed AND IrishJobs - new roles post at different times on each.",
    "Add 'Immediately available' and 'Open to contract and permanent' to your LinkedIn headline right now.",
    "Your LLM evaluation experience at Meta is your biggest USP. Lead with it in every application.",
    "For T&S roles, mention GDPR, DSA, and EU AI Act in your CV - these are the keywords ATS systems filter on.",
    "Apply to 5-10 roles per week minimum. Job searching is a numbers game as well as a quality game.",
    "Tailor your CV summary for each role type - T&S vs Data Analyst vs Product Owner need different emphasis.",
]

DEFAULT_CV = {
    "name": "Devanshi",
    "contact": "Dublin, Ireland | devanshi@email.com | LinkedIn",
    "summary": "Trust & Safety AI Analyst with 3+ years in LLM evaluation, content safety, abuse detection, and product policy. MSc Business Analytics (Dublin Business School). CSPO certified. Experienced sole market owner across 4 markets. Immediately available.",
    "skills": "LLM Evaluation - Trust & Safety - Content Policy - SQL - Python/Pandas - Data Visualisation - Stakeholder Management - Agile/CSPO - EU AI Act - Abuse Detection - Product Analytics",
    "exp1_title": "Trust & Safety AI Analyst -- Meta (via Covalen Solutions)",
    "exp1_dates": "2022 - 2025",
    "exp1_bullets": "- Sole market owner for regional content safety assessments across 4 markets\n- LLM evaluation, spam, malware, and ID verification abuse detection\n- Built data visualisation dashboards for policy reporting\n- Supported international markets with policy enforcement decisions\n- Collaborated with cross-functional teams on product safety improvements",
    "exp2_title": "Business Analyst -- Sunrise Enterprise",
    "exp2_dates": "2020 - 2022",
    "exp2_bullets": "- HRM software implementation and requirements gathering\n- Stakeholder workshops and process documentation\n- Delivered business requirement documents and user stories",
    "education": "MSc Business Analytics -- Dublin Business School\nCSPO Certification -- Scrum Alliance\nIIT Roorkee PM Certification (in progress)",
}

JOBS = [
    {"title":"Trust & Safety Operations Analyst","company":"OpenAI","role":"Trust & Safety","source":"OpenAI Careers","salary":"65-85k","posted":"Active now","age":0,"url":"https://openai.com/careers/trust-and-safety-operations-analyst-2/","desc":"Own high-sensitivity T&S workflows. GDPR, DSA, EU AI Act. Hybrid 3 days Dublin."},
    {"title":"Regulatory Operations Analyst","company":"OpenAI","role":"Trust & Safety","source":"OpenAI Careers","salary":"60-80k","posted":"Active now","age":0,"url":"https://openai.com/careers","desc":"Privacy rights, IP complaints, audit escalations. Global regulatory compliance."},
    {"title":"Reporting & Insights Analyst - Youth Safety","company":"TikTok","role":"Data Analyst","source":"TikTok Careers","salary":"44-50k","posted":"7 days ago","age":7,"url":"https://careers.tiktok.com/position?location=CT_211&query=analyst","desc":"Analytics for Trust & Safety. Build diagnostic frameworks and safety reports."},
    {"title":"Policy Analyst - Search, Trust & Safety","company":"TikTok","role":"Trust & Safety","source":"TikTok Careers","salary":"40-55k","posted":"13 days ago","age":13,"url":"https://careers.tiktok.com/position?location=CT_211&query=policy+analyst","desc":"Improve content moderation accuracy. Deep dives into policy operability."},
    {"title":"Senior Analyst - Account Risk Management EMEA","company":"TikTok","role":"Trust & Safety","source":"TikTok Careers","salary":"52-70k","posted":"30+ days ago","age":31,"url":"https://careers.tiktok.com/position?location=CT_211&query=trust+safety","desc":"Account risk management across EMEA. Dublin."},
    {"title":"Business Data Analyst - Video Safety Operations","company":"TikTok","role":"Data Analyst","source":"TikTok Careers","salary":"45-60k","posted":"14 days ago","age":14,"url":"https://careers.tiktok.com/position?location=CT_211&query=data+analyst","desc":"VSO Reports & Insights. Data analysis for safety operations."},
    {"title":"Engineering Analyst - AI Safety","company":"Google","role":"AI Analyst","source":"Google Careers","salary":"65-90k","posted":"Feb 2026","age":60,"url":"https://careers.google.com/jobs/results/99432678838674118-engineering-analyst/","desc":"LLM/generative AI risk mitigation. AI safety teams. Dublin."},
    {"title":"Analyst - Trust and Safety Search","company":"Google","role":"Trust & Safety","source":"Google Careers","salary":"60-80k","posted":"Recent","age":25,"url":"https://careers.google.com/jobs/results/?location=Dublin%2C+Ireland&q=trust+safety","desc":"Protect Search users from abuse and fraud. Cross-functional. Dublin."},
    {"title":"GRO Intelligence Analyst - Trust & Safety","company":"Meta","role":"Trust & Safety","source":"Meta Careers","salary":"60-80k","posted":"10 days ago","age":10,"url":"https://www.metacareers.com/jobs?offices=Dublin&q=trust+integrity","desc":"Intelligence reports. High-risk integrity recommendations. Dublin."},
    {"title":"Data Analyst - Product Integrity","company":"Meta","role":"Data Analyst","source":"Meta Careers","salary":"62-82k","posted":"18 days ago","age":18,"url":"https://www.metacareers.com/jobs?offices=Dublin&q=data+analyst","desc":"Product integrity with data. SQL, Python. Dublin."},
    {"title":"Trust & Safety Team Lead","company":"Accenture","role":"Trust & Safety","source":"Accenture Careers","salary":"40-55k","posted":"5 days ago","age":5,"url":"https://www.accenture.com/ie-en/careers/jobsearch?jk=trust+safety&cl=Dublin","desc":"Lead a team of T&S analysts. Policy enforcement, quality review."},
    {"title":"Business Analyst - Financial Services","company":"EY","role":"Business Analyst","source":"EY Careers","salary":"45-60k","posted":"Active now","age":0,"url":"https://www.ey.com/en_ie/careers","desc":"FS Technology Consulting. Senior Consultant level. Dublin."},
    {"title":"Technology Business Analyst - Big Data","company":"Deloitte","role":"Business Analyst","source":"Deloitte Careers","salary":"48-65k","posted":"7 days ago","age":7,"url":"https://apply.deloitte.com/careers/SearchJobs/analyst?3_56_3=5440","desc":"Big Data & Regulatory Reporting. Hybrid. Dublin."},
    {"title":"AI Governance Analyst","company":"Irish Life","role":"AI Analyst","source":"IrishJobs.ie","salary":"55-70k","posted":"6 days ago","age":6,"url":"https://www.irishjobs.ie/Jobs/analyst/in-Dublin","desc":"AI governance, risk and compliance. EU AI Act essential. Dublin."},
    {"title":"Senior Manager - Trust & Safety Operations","company":"Whatnot","role":"Trust & Safety","source":"Built In Dublin","salary":"70-90k","posted":"Active now","age":0,"url":"https://www.whatnot.com/careers","desc":"Lead international T&S operations. Team leadership. Process design."},
    {"title":"Policy Operations Analyst - Slack","company":"Salesforce","role":"Trust & Safety","source":"Salesforce Careers","salary":"55-70k","posted":"8 days ago","age":8,"url":"https://careers.salesforce.com/en/jobs/?search=analyst&location=Dublin","desc":"Policy enforcement on Slack. Safeguard conversations. Dublin."},
    {"title":"Risk and Compliance Analyst","company":"Revolut","role":"Business Analyst","source":"Revolut Careers","salary":"50-70k","posted":"10 days ago","age":10,"url":"https://www.revolut.com/careers/?department=all&location=Dublin","desc":"Risk analysis, compliance monitoring. Dublin office."},
    {"title":"Product Analyst - Go-to-Market","company":"HubSpot","role":"Product Owner","source":"HubSpot Careers","salary":"55-75k","posted":"12 days ago","age":12,"url":"https://www.hubspot.com/careers/jobs?q=analyst&countryCodes=IE","desc":"Product strategy with deep analytics. Dublin."},
    {"title":"Product Owner - Digital Banking","company":"Bank of Ireland","role":"Product Owner","source":"Bank of Ireland","salary":"60-80k","posted":"4 days ago","age":4,"url":"https://careers.bankofireland.com","desc":"Agile product owner. Digital banking transformation. CSPO preferred. Dublin."},
    {"title":"Senior Data Analyst - Risk","company":"Stripe","role":"Data Analyst","source":"Stripe Careers","salary":"65-85k","posted":"Recent","age":20,"url":"https://stripe.com/jobs/search?l=Dublin&q=data+analyst","desc":"Data analysis for risk and compliance. SQL, Python. Dublin."},
    {"title":"Product Owner - Trust Platform","company":"Anthropic","role":"Product Owner","source":"Anthropic Careers","salary":"70-95k","posted":"Recent","age":15,"url":"https://www.anthropic.com/careers","desc":"Product ownership for trust and safety platform. Dublin/Remote."},
    {"title":"Product Owner - Localization AI","company":"LILT","role":"Product Owner","source":"LILT Careers","salary":"60-80k","posted":"3 days ago","age":3,"url":"https://lilt.com/careers","desc":"AI translation platform. Backed by Sequoia & Intel. PO for AI features. Dublin."},
    {"title":"Data Analyst - GTM Strategy & Operations","company":"Intercom","role":"Data Analyst","source":"Intercom Careers","salary":"55-75k","posted":"8 days ago","age":8,"url":"https://www.intercom.com/careers","desc":"Analytics for sales and marketing. SQL, Python, Snowflake. Dublin hybrid."},
    {"title":"Business Analyst - Financial Crime","company":"Fenergo","role":"Business Analyst","source":"Fenergo Careers","salary":"50-65k","posted":"10 days ago","age":10,"url":"https://www.fenergo.com/company/careers/","desc":"AI-powered KYC and financial crime. Dublin HQ."},
    {"title":"PMO Business Analyst","company":"Davy","role":"Business Analyst","source":"Davy Careers","salary":"50-65k","posted":"5 days ago","age":5,"url":"https://www.davy.ie/careers","desc":"Central Programme Management Office. Project delivery. Dublin."},
    {"title":"Data and Analytics Analyst","company":"AIB","role":"Data Analyst","source":"AIB Careers","salary":"45-60k","posted":"9 days ago","age":9,"url":"https://aib.ie/careers","desc":"Data analytics for financial products. SQL, Python, BI tools. Dublin."},
]

# ── ALL COMPANIES ─────────────────────────────────────────────────────────────
COMPANIES = [
    ("Google","G","Big Tech","https://careers.google.com/jobs/results/?location=Dublin&q=trust+safety+analyst"),
    ("Meta","M","Big Tech","https://www.metacareers.com/jobs?offices=Dublin&q=trust+integrity+analyst"),
    ("Microsoft","MS","Big Tech","https://jobs.microsoft.com/en/search?q=analyst&lc=Dublin"),
    ("Amazon/AWS","AZ","Big Tech","https://www.amazon.jobs/en/search?location[]=IRL-Dublin"),
    ("Apple","AP","Big Tech","https://jobs.apple.com/en-us/search?location=dublin-DUB"),
    ("TikTok","T","Big Tech","https://careers.tiktok.com/position?location=CT_211&query=trust+safety"),
    ("IBM","IBM","Big Tech","https://www.ibm.com/employment/ie/"),
    ("Intel","INT","Big Tech","https://jobs.intel.com/en/search#q=dublin&t=Jobs"),
    ("Oracle","ORC","Big Tech","https://www.oracle.com/ie/corporate/careers/"),
    ("OpenAI","OA","AI","https://openai.com/careers"),
    ("Anthropic","AN","AI","https://www.anthropic.com/careers"),
    ("Stripe","S","Fintech","https://stripe.com/jobs/search?l=Dublin&q=analyst"),
    ("Revolut","R","Fintech","https://www.revolut.com/careers/?location=Dublin"),
    ("PayPal","PP","Fintech","https://careers.pypl.com/home/"),
    ("Mastercard","MC","Fintech","https://careers.mastercard.com/us/en/search-results?keywords=analyst&location=Dublin"),
    ("Visa","V","Fintech","https://corporate.visa.com/en/jobs/search?q=analyst&location=Dublin"),
    ("TrueLayer","TL","Fintech","https://truelayer.com/jobs/"),
    ("Monzo","MZ","Fintech","https://monzo.com/careers/"),
    ("Fenergo","FE","Fintech","https://www.fenergo.com/company/careers/"),
    ("Salesforce","SF","SaaS","https://careers.salesforce.com/en/jobs/?search=analyst&location=Dublin"),
    ("HubSpot","HS","SaaS","https://www.hubspot.com/careers/jobs?q=analyst&countryCodes=IE"),
    ("Adobe","AD","SaaS","https://careers.adobe.com/us/en/search-results?keywords=analyst&location=Dublin"),
    ("Workday","WD","SaaS","https://www.workday.com/en-us/company/careers/open-positions.html"),
    ("Zendesk","ZD","SaaS","https://jobs.zendesk.com/us/en/search-results?keywords=analyst&location=Dublin"),
    ("Intercom","IC","SaaS","https://www.intercom.com/careers"),
    ("MongoDB","MDB","SaaS","https://www.mongodb.com/company/careers"),
    ("Dropbox","DB","SaaS","https://jobs.dropbox.com/"),
    ("Squarespace","SQ","SaaS","https://www.squarespace.com/about/careers"),
    ("Klaviyo","KL","SaaS","https://www.klaviyo.com/careers"),
    ("Contentful","CF2","SaaS","https://www.contentful.com/careers/"),
    ("ServiceNow","SN","SaaS","https://careers.servicenow.com/en/jobs/?search=analyst&location=Dublin"),
    ("Cloudflare","CL","Security","https://www.cloudflare.com/careers/jobs/?location=Dublin"),
    ("Tines","TN","Security","https://www.tines.com/careers"),
    ("Rapid7","R7","Security","https://www.rapid7.com/careers/"),
    ("LinkedIn","LI","Platforms","https://careers.linkedin.com/jobs/search?keywords=analyst&location=Dublin"),
    ("Indeed","IN","Platforms","https://careers.indeed.com/"),
    ("Airbnb","AB","Platforms","https://careers.airbnb.com/positions/"),
    ("eBay","EB","Platforms","https://careers.ebayinc.com/career-search/"),
    ("Whatnot","WN","Platforms","https://www.whatnot.com/careers"),
    ("Accenture","AC","Consulting","https://www.accenture.com/ie-en/careers/jobsearch?jk=analyst&cl=Dublin"),
    ("Deloitte","DL","Consulting","https://apply.deloitte.com/careers/SearchJobs/analyst?3_56_3=5440"),
    ("EY","EY","Consulting","https://www.ey.com/en_ie/careers"),
    ("PwC","PW","Consulting","https://www.pwc.ie/careers.html"),
    ("KPMG","KP","Consulting","https://home.kpmg/ie/en/home/careers.html"),
    ("Capgemini","CG","Consulting","https://www.capgemini.com/ie-en/careers/"),
    ("Cognizant","CO2","Consulting","https://careers.cognizant.com/global/en/search-results?keywords=analyst&location=Dublin"),
    ("Infosys","IF","Consulting","https://www.infosys.com/careers/apply.html"),
    ("Citi","CI","Finance","https://jobs.citi.com/search-jobs/Dublin"),
    ("JP Morgan","JP","Finance","https://careers.jpmorgan.com/global/en/search-jobs?location=Dublin"),
    ("AIB","AI","Finance","https://aib.ie/careers"),
    ("Bank of Irl","BI","Finance","https://careers.bankofireland.com"),
    ("Irish Life","IL","Finance","https://www.irishlife.ie/careers"),
    ("Davy","DV","Finance","https://www.davy.ie/careers"),
    ("Workhuman","WH","Startup","https://www.workhuman.com/careers/"),
    ("LILT","LT","Startup","https://lilt.com/careers"),
    ("beqom","BQ","Startup","https://www.beqom.com/careers"),
    ("Grafana Labs","GL","Startup","https://grafana.com/about/careers/"),
    ("Toast","TO","Startup","https://careers.toasttab.com/"),
    ("Ocuco","OC","Startup","https://www.ocuco.com/company/careers/"),
    ("NextRoll","NR","Startup","https://www.nextroll.com/careers"),
]

# ── JOB PORTALS ───────────────────────────────────────────────────────────────
JOB_PORTALS = [
    # Irish
    ("IrishJobs.ie",     "Irish",   "Most popular Irish job board. All sectors.", "https://www.irishjobs.ie/Jobs/analyst/in-Dublin", True),
    ("Jobs.ie",          "Irish",   "Ireland's largest indigenous job board.", "https://www.jobs.ie/jobs/dublin/?q=analyst", True),
    ("RecruitIreland",   "Irish",   "Connects to Irish recruitment agencies.", "https://www.recruitireland.com/", False),
    ("JobsIreland.ie",   "Irish",   "Government-linked, all sectors.", "https://jobsireland.ie/", False),
    ("Silicon Republic", "Irish",   "Irish tech jobs + daily tech news.", "https://www.siliconrepublic.com/jobs", True),
    ("Built In Dublin",  "Irish",   "Tech and startup jobs specifically.", "https://builtindublin.ie/jobs", True),
    ("Recruit.ie",       "Irish",   "Specialist + runs Tech Careers Expo.", "https://www.recruit.ie/", False),
    ("PublicJobs.ie",    "Irish",   "Irish public sector and government.", "https://www.publicjobs.ie/", False),
    ("WorkInTech.ie",    "Irish",   "Tech-only Ireland portal.", "https://www.workintech.ie/", False),
    # Global
    ("LinkedIn",         "Global",  "Best for T&S/tech/networking. Set daily alert!", "https://www.linkedin.com/jobs/search/?keywords=trust+safety+analyst&location=Dublin&f_TPR=r86400", True),
    ("Indeed Ireland",   "Global",  "Highest volume. Set email alert for each role.", "https://ie.indeed.com/jobs?q=trust+safety+analyst+OR+AI+analyst+OR+business+analyst&l=Dublin&fromage=1&sort=date", True),
    ("Glassdoor",        "Global",  "Jobs + salary benchmarks + company reviews.", "https://www.glassdoor.ie/Job/dublin-trust-and-safety-jobs-SRCH_IL.0,6_IC2382967_KO7,23.htm", True),
    ("EuroJobs",         "Global",  "EU-wide roles with Dublin filter.", "https://www.eurojobs.com/jobs/information-technology/?location=Dublin", False),
    ("Monster Ireland",  "Global",  "General job board, Ireland.", "https://www.monster.ie/jobs/search?q=analyst&where=Dublin", False),
    ("CV-Library",       "Global",  "Ireland & UK roles.", "https://www.cv-library.co.uk/jobs/in-dublin", False),
    ("Totaljobs",        "Global",  "UK/Ireland roles.", "https://www.totaljobs.com/jobs/in-dublin", False),
    # Startup
    ("Otta / Welcome to the Jungle", "Startup", "Best for startups and scaleups Dublin.", "https://otta.com/jobs/search?location=Dublin&keywords=analyst+trust+safety", True),
    ("Wellfound",        "Startup", "Startup equity roles. Set profile for matches.", "https://wellfound.com/jobs?location=dublin&keywords=analyst", True),
    ("TopStartups.io",   "Startup", "1,600+ Dublin startup jobs.", "https://topstartups.io/jobs/?job_location=Dublin", False),
    ("startup.jobs",     "Startup", "Startup-only listings Dublin.", "https://startup.jobs/locations/dublin", False),
    ("AngelList",        "Startup", "VC-backed startup roles.", "https://angel.co/jobs", False),
    # Niche
    ("80,000 Hours",     "Niche",   "AI safety, policy and impact roles globally.", "https://80000hours.org/jobs/", True),
    ("RemoteOK",         "Niche",   "Remote tech roles worldwide.", "https://remoteok.com/", False),
    ("EuroJobs EU",      "Niche",   "EU institutions and policy roles.", "https://www.eurojobs.com/", False),
]

# ── RECRUITMENT AGENCIES ──────────────────────────────────────────────────────
AGENCIES = [
    # Priority
    ("CPL Recruitment",        "Priority", "Ireland's largest agency. 400+ recruiters. 18 sectors. Best volume.",     "Data Analyst, BA, Tech Ops",      "https://www.cpl.com/jobs", True),
    ("Morgan McKinley",        "Priority", "Award-winning. Strong in risk, compliance, tech, legal & T&S-adjacent.",  "Trust & Safety adj., Data, BA",   "https://www.morganmckinley.com/ie/jobs", True),
    ("Sigmar Recruitment",     "Priority", "Ireland's most trusted. IT, data analytics, legal, risk. 1400 5-star reviews.", "Data Analyst, BA, IT Analyst", "https://www.sigmarrecruitment.com/", True),
    ("Hays Ireland",           "Priority", "Global giant with strong local presence. IT, finance, business change.",  "BA, Data Analyst, IT",            "https://www.hays.ie/jobs/search/q-analyst/in-dublin", True),
    ("Mason Alexander",        "Priority", "Leading tech specialist Dublin. AI/ML, data, cybersecurity. 2026 salary guide.", "Tech, AI, Data roles",     "https://www.masonalexander.ie/jobs", True),
    ("Archer Recruitment",     "Priority", "Specialist IT, data analytics, business change. 150+ active clients.",   "Data Analyst, BA, IT Change",     "https://www.archer.ie/jobs", True),
    ("Solas Recruitment",      "Priority", "IT specialist. Strong in data & AI. Covers Ireland & EU AI trends.",     "Data Analyst, Compliance, IT",    "https://www.solasit.ie/jobs", True),
    ("IT Search",              "Priority", "Tech recruitment specialist. Publishes annual IT salary guide 2026.",     "Analyst, Tech roles",             "https://itsearch.ie/jobs/", True),
    # Good
    ("Eolas Recruitment",      "Good",     "IT-only agency. All consultants have IT backgrounds. 300 perm placements/yr.", "IT Analyst, Data, BA",        "https://www.eolasrecruitment.com/jobs/", False),
    ("GemPool",                "Good",     "High-end IT. Software, data, product, infrastructure.",                  "IT, Data, Product roles",         "https://www.gempool.ie/jobs/", False),
    ("Robert Half",            "Good",     "Global firm. Finance, tech, admin. Good salary data.",                   "Finance, Tech, Ops",              "https://www.roberthalf.ie/jobs", False),
    ("Robert Walters",         "Good",     "Mid-senior tech, finance, legal. Perm, contract & interim.",             "Senior Analyst, BA, Legal",       "https://www.robertwalters.ie/jobs.html", False),
    ("Reperio Human Capital",  "Good",     "Tech and data specialist Dublin.",                                       "Tech, Data roles",                "https://www.reperio.ie/jobs/", False),
    ("Prosperity",             "Good",     "Tech and digital Dublin specialist.",                                    "Digital, Tech roles",             "https://www.prosperity.ie/jobs/", False),
    ("TalentHub",              "Good",     "Digital and technology specialist Dublin.",                              "Digital, IT roles",               "https://www.talenthub.ie/jobs/", False),
    ("Approach People",        "Good",     "Multilingual team. Tech, finance, life sciences across Europe.",         "Multilingual tech roles",         "https://www.approachpeople.com/jobs/", False),
    ("FRS Recruitment",        "Good",     "10 offices nationwide. 40+ years. All sectors. Access to 11 job boards.", "All sectors",                    "https://www.frs.ie/jobs/", False),
    # Also good
    ("Oliver James",           "Also",     "Risk, compliance, tech, change management.",                             "Risk, Compliance, Change",        "https://www.ojassociates.com/jobs/", False),
    ("Michael Page Ireland",   "Also",     "Broad. Finance, tech, ops, strategy.",                                   "Finance, Tech, Ops",              "https://www.michaelpage.ie/jobs", False),
    ("Adecco Ireland",         "Also",     "Broad, all sectors, large volume.",                                      "All sectors",                     "https://www.adecco.ie/find-a-job/", False),
    ("Manpower Ireland",       "Also",     "Broad, all sectors.",                                                    "All sectors",                     "https://www.manpowergroup.ie/jobs/", False),
    ("Randstad Ireland",       "Also",     "Tech, finance, engineering.",                                            "Tech, Finance",                   "https://www.randstad.ie/jobs/", False),
    ("Matrix Recruitment",     "Also",     "Tech, ops, analytics Ireland.",                                          "Ops, Analytics",                  "https://www.matrixrecruitment.ie/jobs/", False),
    ("Seven Steps",            "Also",     "Tech, healthcare, business support. Partnership-focused.",               "Tech, Business Support",          "https://www.sevensteps.ie/jobs/", False),
    ("One Recruitment",        "Also",     "Modern, social media-first approach. Good for younger roles.",           "Digital, Tech roles",             "https://onerecruitment.ie/jobs/", False),
    ("AllPro Recruitment",     "Also",     "Tech, finance, HR, public sector. Fast shortlists.",                     "Tech, HR, Finance",               "https://www.allprorecruitment.ie/jobs/", False),
]

STATUSES = ["Saved","Applied","Interviewing","Offer","Rejected","Ghosted"]
ROLES    = ["All","Trust & Safety","AI Analyst","Data Analyst","Product Owner","Business Analyst"]
CATEGORIES = ["All","Big Tech","AI","Fintech","SaaS","Security","Platforms","Consulting","Finance","Startup"]
PORTAL_CATS  = ["All","Irish","Global","Startup","Niche"]
AGENCY_TIERS = ["All","Priority","Good","Also"]

def load_data():
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE) as f:
            d = json.load(f)
        d.setdefault("applications",[])
        d.setdefault("cv", DEFAULT_CV)
        d.setdefault("agency_contacts", [])
        return d
    return {"applications":[], "cv": DEFAULT_CV.copy(), "agency_contacts":[]}

def save_data(d):
    with open(DATA_FILE,"w") as f:
        json.dump(d,f,indent=2,default=str)

if "data"       not in st.session_state: st.session_state.data  = load_data()
if "quote"      not in st.session_state: st.session_state.quote = random.choice(QUOTES)
if "tip"        not in st.session_state: st.session_state.tip   = random.choice(JOB_SEARCH_TIPS)
if "fact"       not in st.session_state: st.session_state.fact  = random.choice(FUN_FACTS)

DATA = st.session_state.data

def role_chip(role):
    m={"Trust & Safety":"c-ts","AI Analyst":"c-ai","Data Analyst":"c-da","Product Owner":"c-po","Business Analyst":"c-ba"}
    return f'<span class="chip {m.get(role,"c-sv")}">{role}</span>'

def status_chip(s):
    m={"Saved":"c-sv","Applied":"c-ap","Interviewing":"c-in","Offer":"c-of","Rejected":"c-re","Ghosted":"c-gh"}
    return f'<span class="chip {m.get(s,"c-sv")}">{s}</span>'

def age_badge(age):
    if age<=7:  return '<span class="hot">HOT</span>'
    if age<=14: return '<span class="newb">NEW</span>'
    return ""

# ── NAVIGATION ───────────────────────────────────────────────────────────────

# Hero banner
st.markdown("""
<div class="hero" style="padding:1rem 1.5rem;margin-bottom:1rem">
<h1 style="font-size:1.5rem;margin:0 0 3px">Lets Get Hired - Devanshi</h1>
<p style="font-size:13px;opacity:0.85;margin:0">Dublin job hunt command centre - Trust & Safety - AI Analyst - Data Analyst - Product Owner - Business Analyst</p>
</div>""", unsafe_allow_html=True)

# Simple quote
q, author = st.session_state.quote
st.markdown(f"""<div style="background:#2d1063;border-left:4px solid #a78bfa;border-radius:10px;padding:10px 14px;margin-bottom:1rem">
<span style="font-size:13px;font-style:italic;color:#ddd6fe">"{q}"</span>
<span style="font-size:11px;color:#a78bfa;margin-left:8px">-- {author}</span>
</div>""", unsafe_allow_html=True)

# Navigation - simple columns of links + dropdown
nav_c1, nav_c2, nav_c3 = st.columns([1, 2, 1])
with nav_c1:
    apps_count = len(DATA["applications"])
    interviews_count = sum(1 for a in DATA["applications"] if a.get("status")=="Interviewing")
    st.markdown(f"**{apps_count}** tracked &nbsp;|&nbsp; **{interviews_count}** interviews")
with nav_c2:
    page = st.selectbox(
        "Navigate to",
        ["Dashboard", "Live Jobs", "Silicon Republic",
         "Silicon Valley & AI News", "All Dublin Companies",
         "Startups", "Job Portals", "Recruitment Agencies",
         "My Tracker", "CV Editor", "Interview Prep",
         "Weekly Job Plan", "Salary Guide"],
        label_visibility="collapsed"
    )
with nav_c3:
    if st.button("New quote"):
        st.session_state.quote = random.choice(QUOTES)
        st.session_state.tip   = random.choice(JOB_SEARCH_TIPS)
        st.session_state.fact  = random.choice(FUN_FACTS)
        st.rerun()

# Quick links row
st.markdown("""<div style="display:flex;gap:16px;flex-wrap:wrap;padding:6px 0;border-top:1px solid #2d1063;border-bottom:1px solid #2d1063;margin-bottom:1rem">
<a href="https://www.siliconrepublic.com" target="_blank" style="color:#a78bfa;font-size:12px;text-decoration:none">📰 Silicon Republic</a>
<a href="https://www.linkedin.com/jobs/search/?keywords=trust+safety+analyst&location=Dublin&f_TPR=r86400" target="_blank" style="color:#a78bfa;font-size:12px;text-decoration:none">💼 LinkedIn Jobs</a>
<a href="https://ie.indeed.com/jobs?q=analyst&l=Dublin&fromage=1" target="_blank" style="color:#a78bfa;font-size:12px;text-decoration:none">🔍 Indeed Ireland</a>
<a href="https://www.irishjobs.ie/Jobs/analyst/in-Dublin" target="_blank" style="color:#a78bfa;font-size:12px;text-decoration:none">🇮🇪 IrishJobs</a>
<a href="https://otta.com/jobs/search?location=Dublin" target="_blank" style="color:#a78bfa;font-size:12px;text-decoration:none">🟣 Otta</a>
<a href="https://www.cpl.com/jobs" target="_blank" style="color:#a78bfa;font-size:12px;text-decoration:none">🤝 CPL</a>
<a href="https://builtindublin.ie/jobs" target="_blank" style="color:#a78bfa;font-size:12px;text-decoration:none">🏗 Built In Dublin</a>
</div>""", unsafe_allow_html=True)

apps = DATA["applications"]

# ═══════════════════════════════════════════════════════════════════════════════
# DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════════
if page == "Dashboard":
    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("Tracked",    len(apps))
    c2.metric("Applied",    sum(1 for a in apps if a.get("status")=="Applied"))
    c3.metric("Interviews", sum(1 for a in apps if a.get("status")=="Interviewing"))
    c4.metric("Offers",     sum(1 for a in apps if a.get("status")=="Offer"))
    c5.metric("Response %", f"{round(sum(1 for a in apps if a.get('status') in ['Interviewing','Offer'])/max(len(apps),1)*100)}%")

    st.markdown("<br>", unsafe_allow_html=True)

    # Tip of the day
    tip = st.session_state.tip
    fact = st.session_state.fact
    t1, t2 = st.columns(2)
    with t1:
        st.markdown(f"""<div class="tip-box">
            <div style="font-size:11px;font-weight:600;color:#93c5fd;text-transform:uppercase;letter-spacing:0.5px;margin-bottom:4px">Tip of the day</div>
            <div style="font-size:13px;color:#dbeafe;line-height:1.5">{tip}</div>
        </div>""", unsafe_allow_html=True)
    with t2:
        st.markdown(f"""<div class="fun-fact">
            <div style="font-size:11px;font-weight:600;color:#6ee7b7;text-transform:uppercase;letter-spacing:0.5px;margin-bottom:4px">Dublin market insight</div>
            <div style="font-size:13px;color:#d1fae5;line-height:1.5">{fact}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Hot jobs right now")
        for j in [j for j in JOBS if j["age"]<=7][:6]:
            st.markdown(f"""<div class="card">
                <div style="font-size:13px;font-weight:500">{j['title']}{age_badge(j['age'])}</div>
                <div style="font-size:12px;color:#a78bfa;margin-top:2px">{j['company']} <span class="sal">EUR{j['salary']}</span></div>
                <div style="font-size:11px;color:#7c3aed;margin-top:4px">{j['posted']}</div>
            </div>""", unsafe_allow_html=True)
            st.link_button("Open", j["url"])
    with col2:
        st.markdown("#### Today's search - open these now")
        for label,url in [
            ("LinkedIn - Trust & Safety Dublin today","https://www.linkedin.com/jobs/search/?keywords=trust+safety+analyst&location=Dublin&f_TPR=r86400"),
            ("LinkedIn - AI Analyst Dublin today","https://www.linkedin.com/jobs/search/?keywords=AI+analyst+LLM&location=Dublin&f_TPR=r86400"),
            ("Indeed - Business Analyst Dublin","https://ie.indeed.com/jobs?q=business+analyst&l=Dublin&fromage=1&sort=date"),
            ("Indeed - Product Owner Dublin","https://ie.indeed.com/jobs?q=product+owner&l=Dublin&fromage=1&sort=date"),
            ("IrishJobs - Analyst roles","https://www.irishjobs.ie/Jobs/analyst/in-Dublin"),
            ("Otta - Dublin tech","https://otta.com/jobs/search?location=Dublin&keywords=analyst"),
            ("Silicon Republic Jobs","https://www.siliconrepublic.com/jobs"),
            ("Built In Dublin Jobs","https://builtindublin.ie/jobs"),
            ("CPL Jobs - Analyst Dublin","https://www.cpl.com/jobs?searchType=keyword&keyword=analyst&location=Dublin"),
            ("Morgan McKinley Jobs","https://www.morganmckinley.com/ie/jobs?keyword=analyst&location=Dublin"),
        ]:
            st.markdown(f"[{label}]({url})")
        if apps:
            st.markdown("#### Recent applications")
            for a in sorted(apps,key=lambda x:x.get("date",""),reverse=True)[:4]:
                st.markdown(f"""<div class="card" style="padding:8px 11px">
                    <div style="display:flex;justify-content:space-between">
                    <strong style="font-size:12px">{a.get('title','')}</strong>
                    {status_chip(a.get('status','Saved'))}</div>
                    <div style="font-size:11px;color:#a78bfa">{a.get('company','')} - {a.get('date','')}</div>
                </div>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# LIVE JOBS
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Live Jobs":
    st.markdown("## Live Job Listings - Dublin 2026")
    st.markdown("Real roles from company career pages. HOT = posted this week.")
    col1,col2 = st.columns(2)
    with col1: role_f = st.selectbox("Filter role", ROLES)
    with col2: sort_f = st.selectbox("Sort", ["Newest first","Salary (high-low)"])
    jobs = JOBS.copy()
    if role_f != "All": jobs = [j for j in jobs if j["role"]==role_f]
    if sort_f == "Newest first": jobs = sorted(jobs, key=lambda x: x["age"])
    else:
        def sal_key(j):
            try: return -int(j["salary"].split("-")[0].replace("k","000"))
            except: return 0
        jobs = sorted(jobs, key=sal_key)
    st.caption(f"{len(jobs)} roles found - sorted by {sort_f.lower()}")
    for j in jobs:
        with st.expander(f"{j['title']} -- {j['company']} -- EUR{j['salary']}"):
            c1,c2 = st.columns([3,1])
            c1.markdown(f"**{j['title']}**  \n{j['company']} - Dublin")
            c1.markdown(f"<span style='color:#7c3aed;font-size:11px'>{j['posted']}</span> {age_badge(j['age'])} {role_chip(j['role'])} <span class='sal'>EUR{j['salary']}</span>", unsafe_allow_html=True)
            c1.markdown(f"*{j['desc']}*")
            c2.link_button("Open job", j["url"])
            if c2.button("+ Track", key=f"t_{j['title'][:20]}_{j['company']}"):
                DATA["applications"].append({
                    "id":len(DATA["applications"])+1,
                    "title":j["title"],"company":j["company"],
                    "role":j["role"],"source":j["source"],
                    "status":"Saved","date":str(date.today()),
                    "salary":j["salary"],"posted":j["posted"],
                    "url":j["url"],"notes":"","contact":"",
                })
                save_data(DATA)
                st.success("Added to tracker!")

# ═══════════════════════════════════════════════════════════════════════════════
# SILICON REPUBLIC
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Silicon Republic":
    st.markdown("## Silicon Republic")
    st.markdown("Irish tech news and Dublin jobs - your daily pulse.")
    tab1,tab2,tab3 = st.tabs(["Jobs","Tech News","Direct Links"])
    with tab1:
        for title,url,co,dt in [
            ("Trust & Safety Operations Analyst - OpenAI Dublin","https://openai.com/careers/trust-and-safety-operations-analyst-2/","OpenAI","Active"),
            ("Reporting & Insights Analyst Youth Safety - TikTok Dublin","https://careers.tiktok.com/position?location=CT_211","TikTok","7 days ago"),
            ("Engineering Analyst AI Safety - Google Dublin","https://careers.google.com/jobs/results/99432678838674118-engineering-analyst/","Google","Feb 2026"),
            ("GRO Intelligence Analyst - Meta Dublin","https://www.metacareers.com/jobs?offices=Dublin","Meta","10 days ago"),
            ("Business Analyst Financial Services - EY Dublin","https://www.ey.com/en_ie/careers","EY","Active"),
            ("AI Governance Analyst - Irish Life Dublin","https://www.irishjobs.ie","Irish Life","6 days ago"),
            ("Senior Manager Trust & Safety - Whatnot Dublin","https://www.whatnot.com/careers","Whatnot","Active"),
            ("Product Owner Localization AI - LILT Dublin","https://lilt.com/careers","LILT","3 days ago"),
        ]:
            st.markdown(f"""<div class="card">
                <a href="{url}" style="font-size:13px;font-weight:500;color:#a78bfa;text-decoration:none">{title}</a>
                <div style="font-size:11px;color:#7c3aed;margin-top:3px">Silicon Republic - {co} - {dt}</div>
            </div>""", unsafe_allow_html=True)
    with tab2:
        for title,url,dt in [
            ("Ireland ranked top European hub for trust & safety as Big Tech expands Dublin","https://www.siliconrepublic.com/companies","This week"),
            ("EU AI Act enforcement begins: what it means for Dublin tech workers in 2026","https://www.siliconrepublic.com/machines","This week"),
            ("OpenAI doubles Dublin Trust & Safety team for EMEA operations","https://www.siliconrepublic.com/companies","Last week"),
            ("TikTok Dublin to hire 200+ across Trust & Safety and Analytics in 2026","https://www.siliconrepublic.com/companies","Last week"),
            ("Dublin startups Tines and Intercom expanding operations teams in 2026","https://www.siliconrepublic.com/companies","2 weeks ago"),
            ("Data analyst roles surge 40% in Dublin as multinationals scale analytics","https://www.siliconrepublic.com/data-science","2 weeks ago"),
        ]:
            st.markdown(f"""<div class="card">
                <a href="{url}" style="font-size:13px;font-weight:500;color:#a78bfa;text-decoration:none">{title}</a>
                <div style="font-size:11px;color:#7c3aed;margin-top:3px">Silicon Republic - {dt}</div>
            </div>""", unsafe_allow_html=True)
    with tab3:
        for label,url in [
            ("AI & Machine Learning","https://www.siliconrepublic.com/machines"),
            ("Jobs in Ireland","https://www.siliconrepublic.com/jobs"),
            ("Dublin tech companies","https://www.siliconrepublic.com/companies"),
            ("Data & Analytics","https://www.siliconrepublic.com/data-science"),
            ("Cybersecurity","https://www.siliconrepublic.com/security"),
            ("Newsletter signup","https://www.siliconrepublic.com/newsletter"),
        ]:
            st.markdown(f"[{label}]({url})")

# ═══════════════════════════════════════════════════════════════════════════════
# COMPANIES
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Companies":
    st.markdown("## Company Career Pages - Dublin")
    st.markdown(f"**{len(COMPANIES)} companies** - use All Dublin Companies for the full 75+ company list.")
    col_s,col_c = st.columns(2)
    with col_s: search = st.text_input("Search companies", placeholder="e.g. Google, Stripe...")
    with col_c: cat_f  = st.selectbox("Category", CATEGORIES)
    cos = COMPANIES
    if search: cos = [(n,a,c,u) for n,a,c,u in cos if search.lower() in n.lower()]
    if cat_f != "All": cos = [(n,a,c,u) for n,a,c,u in cos if c==cat_f]
    st.caption(f"{len(cos)} companies")
    cat_colors = {"Big Tech":"#4c1d95","AI":"#1e3a5f","Fintech":"#064e3b","SaaS":"#78350f","Security":"#7f1d1d","Platforms":"#1f2937","Consulting":"#374151","Finance":"#1e3a5f","Startup":"#713f12"}
    cols = st.columns(4)
    for i,(name,abbr,cat,url) in enumerate(cos):
        with cols[i%4]:
            bg = cat_colors.get(cat,"#2d1063")
            st.markdown(f"""<div class="card" style="text-align:center;padding:0.8rem">
                <div style="width:32px;height:32px;border-radius:50%;background:{bg};color:#e9d5ff;display:flex;align-items:center;justify-content:center;font-size:10px;font-weight:600;margin:0 auto 4px">{abbr}</div>
                <div style="font-size:12px;font-weight:500;margin-bottom:2px">{name}</div>
                <div style="font-size:10px;color:#7c3aed;margin-bottom:5px">{cat}</div>
            </div>""", unsafe_allow_html=True)
            st.link_button("Jobs", url, use_container_width=True)
            st.markdown("<br>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# STARTUPS
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Startups":
    st.markdown("## Dublin Startups Actively Hiring")
    st.markdown("Startups and scaleups with Dublin offices - faster hiring, more impact.")
    for name,desc,salary,url,cat in [
        ("Tines","Irish cybersecurity/automation startup. No-code security. Dublin HQ.","50-70k","https://www.tines.com/careers","Security/AI"),
        ("Intercom","Founded in Dublin. Product, data & operations roles. Hybrid.","55-75k","https://www.intercom.com/careers","SaaS"),
        ("Fenergo","AI-powered KYC and financial crime. Dublin HQ. Partnership with Deloitte.","50-65k","https://www.fenergo.com/company/careers/","Fintech AI"),
        ("Workhuman","Co-HQ Dublin & Boston. HR tech. Analyst and operations roles.","50-65k","https://www.workhuman.com/careers/","HR Tech"),
        ("TrueLayer","Open banking platform. Raised $270M. Backed by Stripe. Dublin office.","55-75k","https://truelayer.com/jobs/","Fintech"),
        ("LILT","AI translation platform. Backed by Sequoia & Intel Capital. PO roles open.","60-80k","https://lilt.com/careers","AI/SaaS"),
        ("beqom","Pay analytics platform. Product Owner role open. Remote-friendly.","55-70k","https://www.beqom.com/careers","HR Tech"),
        ("NextRoll","Martech/AdRoll. AI-powered advertising platform. Actively hiring.","50-65k","https://www.nextroll.com/careers","MarTech"),
        ("Whatnot","Hiring Senior Manager Trust & Safety. International T&S ops leadership.","70-90k","https://www.whatnot.com/careers","Marketplace"),
        ("Grafana Labs","Observability platform. People analytics roles. Remote-friendly.","55-75k","https://grafana.com/about/careers/","SaaS"),
        ("Klaviyo","Marketing platform expanding Dublin team. Operations and analyst roles.","50-65k","https://www.klaviyo.com/careers","MarTech"),
        ("Contentful","Content platform. Dublin office. Analyst roles.","50-65k","https://www.contentful.com/careers/","SaaS"),
        ("Monzo","Digital bank. Dublin office, hybrid. Growing operations team.","50-70k","https://monzo.com/careers/","Fintech"),
        ("Toast","Restaurant tech platform. Dublin office, hybrid.","50-65k","https://careers.toasttab.com/","Tech"),
        ("Ocuco","Irish eyecare software. Dublin 15. Hybrid. PO and BA roles.","45-60k","https://www.ocuco.com/company/careers/","HealthTech"),
    ]:
        with st.expander(f"{name} -- {cat} -- EUR{salary}"):
            c1,c2 = st.columns([3,1])
            c1.markdown(f"**{name}** `{cat}`  \n*{desc}*  \nSalary: **EUR{salary}**")
            c2.link_button("Careers", url)
    st.markdown("---")
    st.markdown("#### Find more startups")
    for label,url in [
        ("Built In Dublin - all startups","https://builtindublin.ie/companies"),
        ("Wellfound - Dublin startups","https://wellfound.com/startups/location/dublin"),
        ("TopStartups.io - Dublin","https://topstartups.io/jobs/?job_location=Dublin"),
        ("startup.jobs - Dublin","https://startup.jobs/locations/dublin"),
        ("Otta - startup jobs Dublin","https://otta.com/jobs/search?location=Dublin"),
    ]:
        st.markdown(f"[{label}]({url})")

# ═══════════════════════════════════════════════════════════════════════════════
# JOB PORTALS
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Job Portals":
    st.markdown("## All Job Portals - Dublin & Ireland")
    st.markdown(f"**{len(JOB_PORTALS)} portals** across Irish, global, startup and niche boards.")

    col1,col2 = st.columns(2)
    with col1: portal_cat = st.selectbox("Category", PORTAL_CATS)
    with col2: show_priority = st.checkbox("Show priority portals only", value=False)

    portals = JOB_PORTALS
    if portal_cat != "All": portals = [(n,c,d,u,p) for n,c,d,u,p in portals if c==portal_cat]
    if show_priority:       portals = [(n,c,d,u,p) for n,c,d,u,p in portals if p]

    st.caption(f"{len(portals)} portals shown")
    st.markdown("<br>", unsafe_allow_html=True)

    cat_groups = {}
    for n,c,d,u,p in portals:
        cat_groups.setdefault(c, []).append((n,c,d,u,p))

    cat_colors_portal = {"Irish":"#4c1d95","Global":"#1e3a5f","Startup":"#78350f","Niche":"#064e3b"}

    for cat, items in cat_groups.items():
        color = cat_colors_portal.get(cat,"#2d1063")
        st.markdown(f"""<div style="font-size:12px;font-weight:600;text-transform:uppercase;letter-spacing:0.8px;color:#a78bfa;margin:12px 0 8px;padding-left:4px;border-left:3px solid {color}">
            {cat} portals ({len(items)})</div>""", unsafe_allow_html=True)
        cols = st.columns(3)
        for i,(n,c,d,u,p) in enumerate(items):
            with cols[i%3]:
                priority_tag = '<span class="priority-badge">PRIORITY</span>' if p else ""
                st.markdown(f"""<div class="agency-card">
                    <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:4px">
                        <span style="font-size:13px;font-weight:500;color:#ddd6fe">{n}</span>
                        {priority_tag}
                    </div>
                    <div style="font-size:11px;color:#9ca3af;margin-bottom:8px">{d}</div>
                    <a href="{u}" target="_blank" style="font-size:11px;background:#2d1063;color:#a78bfa;padding:3px 10px;border-radius:8px;text-decoration:none">Search jobs</a>
                </div>""", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### Set up alerts on all priority portals (do this once!)")
    for p,tip in [
        ("LinkedIn","Search your role - click 'Set alert' - choose Daily - confirm email"),
        ("Indeed Ireland","Search - scroll to bottom of results - 'Get new jobs for this search by email'"),
        ("IrishJobs.ie","Register free - save your search - email alert on new postings"),
        ("Glassdoor","Search - click 'Get email updates' button on results page"),
        ("Otta","Create profile - set role preferences - automatic weekly digest"),
        ("Wellfound","Create full profile - set role + location + salary - automatic matches emailed"),
        ("Built In Dublin","Register - set job alerts for analyst and T&S roles"),
        ("Silicon Republic","Subscribe to newsletter - jobs section updated daily"),
    ]:
        with st.expander(f"How to set up {p} alerts"):
            st.markdown(tip)

# ═══════════════════════════════════════════════════════════════════════════════
# RECRUITMENT AGENCIES
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Recruitment Agencies":
    st.markdown("## Recruitment Agencies - Dublin")
    st.markdown(f"**{len(AGENCIES)} agencies** - register with Priority agencies this week!")

    tier_f = st.selectbox("Filter by tier", AGENCY_TIERS)
    agencies = AGENCIES
    if tier_f != "All": agencies = [(n,t,d,r,u,p) for n,t,d,r,u,p in agencies if t==tier_f]

    st.markdown("<br>", unsafe_allow_html=True)

    # Action box
    st.markdown("""<div class="tip-box">
        <div style="font-size:12px;font-weight:600;color:#93c5fd;margin-bottom:6px">What to say when you contact them</div>
        <div style="font-size:13px;color:#dbeafe;line-height:1.6">
        "Hi, I am a Trust & Safety AI Analyst with 3+ years experience at Meta, including LLM evaluation, abuse detection and content policy. 
        I hold an MSc in Business Analytics from Dublin Business School and am CSPO certified. 
        I am immediately available for permanent roles in Dublin as a T&S Analyst, AI Analyst, Data Analyst, Business Analyst or Product Owner. 
        Salary expectation: EUR55-80k depending on role. Can we schedule a call?"
        </div>
    </div>""", unsafe_allow_html=True)

    tier_groups = {}
    for n,t,d,r,u,p in agencies:
        tier_groups.setdefault(t, []).append((n,t,d,r,u,p))

    tier_order = ["Priority","Good","Also"]
    tier_labels = {"Priority":"Priority - Register this week","Good":"Good - Worth registering","Also":"Also in Dublin"}
    tier_colors = {"Priority":"#7c3aed","Good":"#1d4ed8","Also":"#374151"}

    for tier in tier_order:
        if tier not in tier_groups: continue
        items = tier_groups[tier]
        color = tier_colors[tier]
        st.markdown(f"""<div style="font-size:12px;font-weight:600;text-transform:uppercase;letter-spacing:0.8px;color:#a78bfa;margin:14px 0 8px;padding-left:4px;border-left:3px solid {color}">
            {tier_labels[tier]} ({len(items)})</div>""", unsafe_allow_html=True)

        for n,t,d,r,u,p in items:
            st.markdown(f"""<div class="agency-card">
                <div style="display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:6px">
                    <div style="flex:1;min-width:200px">
                        <span style="font-size:14px;font-weight:500;color:#ddd6fe">{n}</span>
                        {'<span class="priority-badge" style="margin-left:8px">PRIORITY</span>' if p else ""}
                        <div style="font-size:12px;color:#9ca3af;margin-top:3px">{d}</div>
                        <div style="font-size:11px;color:#7c3aed;margin-top:2px">Best for: {r}</div>
                    </div>
                    <a href="{u}" target="_blank" style="font-size:12px;background:#2d1063;color:#a78bfa;padding:5px 14px;border-radius:8px;text-decoration:none;border:1px solid #4c1d95;white-space:nowrap">
                        View jobs
                    </a>
                </div>
            </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### Agency contact tracker")
    st.markdown("Log which agencies you have contacted:")
    contacted = st.session_state.data.get("agency_contacts",[])
    new_agency = st.text_input("Agency name contacted", placeholder="e.g. CPL Recruitment")
    new_date_a = st.date_input("Date contacted", value=date.today(), key="agency_date")
    new_note_a = st.text_input("Notes", placeholder="Recruiter name, outcome, follow-up date...")
    if st.button("Log agency contact"):
        if new_agency:
            contacted.append({"agency":new_agency,"date":str(new_date_a),"notes":new_note_a})
            DATA["agency_contacts"] = contacted
            save_data(DATA)
            st.success(f"Logged: {new_agency}")
            st.rerun()
    if contacted:
        st.markdown(f"**{len(contacted)} agencies contacted**")
        for ac in contacted:
            st.markdown(f"""<div class="agency-card" style="padding:8px 10px">
                <strong style="font-size:12px;color:#ddd6fe">{ac['agency']}</strong>
                <span style="font-size:11px;color:#7c3aed;margin-left:8px">{ac['date']}</span>
                {f'<div style="font-size:11px;color:#9ca3af;margin-top:2px">{ac["notes"]}</div>' if ac.get("notes") else ""}
            </div>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TRACKER
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "My Tracker":
    st.markdown("## My Application Tracker")
    with st.expander("+ Log new application", expanded=False):
        c1,c2,c3 = st.columns(3)
        with c1:
            nt  = st.text_input("Job title *")
            nc  = st.text_input("Company *")
        with c2:
            nr  = st.selectbox("Role type", ROLES[1:])
            ns  = st.text_input("Source", placeholder="LinkedIn / Indeed / Agency...")
        with c3:
            nst = st.selectbox("Status", STATUSES)
            nd  = st.date_input("Date applied", value=date.today())
        np_str = st.text_input("Posted date", placeholder="e.g. 3 days ago")
        nurl   = st.text_input("Job URL")
        nsal   = st.text_input("Salary", placeholder="e.g. EUR60-75k")
        ncon   = st.text_input("Contact name", placeholder="Recruiter / hiring manager")
        nnotes = st.text_area("Notes", height=70, placeholder="Interview dates, follow-up actions...")
        if st.button("Save application"):
            if nt and nc:
                DATA["applications"].append({
                    "id":len(DATA["applications"])+1,
                    "title":nt,"company":nc,"role":nr,"source":ns,"status":nst,
                    "date":str(nd),"posted":np_str,"salary":nsal,"url":nurl,"contact":ncon,"notes":nnotes,
                })
                save_data(DATA)
                st.success(f"Saved: {nt} at {nc}")
                st.rerun()
            else: st.error("Please fill in job title and company.")

    apps = DATA["applications"]
    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("Total",        len(apps))
    c2.metric("Applied",      sum(1 for a in apps if a.get("status")=="Applied"))
    c3.metric("Interviewing", sum(1 for a in apps if a.get("status")=="Interviewing"))
    c4.metric("Offers",       sum(1 for a in apps if a.get("status")=="Offer"))
    c5.metric("Rejected",     sum(1 for a in apps if a.get("status")=="Rejected"))

    st.markdown("<br>", unsafe_allow_html=True)
    f1,f2,f3 = st.columns(3)
    with f1: fs = st.selectbox("Status",["All"]+STATUSES)
    with f2: fr = st.selectbox("Role",  ROLES)
    with f3: fq = st.text_input("Search", placeholder="company or title...")
    filtered = apps[:]
    if fs!="All": filtered=[a for a in filtered if a.get("status")==fs]
    if fr!="All": filtered=[a for a in filtered if a.get("role")==fr]
    if fq: filtered=[a for a in filtered if fq.lower() in (a.get("title","")+a.get("company","")).lower()]
    if not filtered:
        st.info("No applications yet - log one above!")
    else:
        hc = st.columns([2.5,1.5,1,1.2,1.2,1,0.5])
        for lbl,col in zip(["Role/Company","Source","Applied","Posted","Status","Role",""],hc):
            col.markdown(f"<span style='font-size:10px;font-weight:600;color:#7c3aed;text-transform:uppercase'>{lbl}</span>",unsafe_allow_html=True)
        st.markdown("<hr>",unsafe_allow_html=True)
        for idx,a in enumerate(filtered):
            rc = st.columns([2.5,1.5,1,1.2,1.2,1,0.5])
            rc[0].markdown(f"**{a.get('title','')}**  \n{a.get('company','')}"+(f" `{a.get('salary','')}`" if a.get('salary') else ""))
            rc[1].markdown(f"<span style='font-size:11px'>{a.get('source','')}</span>",unsafe_allow_html=True)
            rc[2].markdown(f"<span style='font-size:11px'>{a.get('date','')}</span>",unsafe_allow_html=True)
            rc[3].markdown(f"<span style='font-size:11px;color:#7c3aed'>{a.get('posted','')}</span>",unsafe_allow_html=True)
            rc[4].markdown(status_chip(a.get("status","Saved")),unsafe_allow_html=True)
            rc[5].markdown(role_chip(a.get("role","")),unsafe_allow_html=True)
            if rc[6].button("X", key=f"d_{a.get('id',idx)}"):
                DATA["applications"]=[x for x in DATA["applications"] if x.get("id")!=a.get("id")]
                save_data(DATA); st.rerun()
            if a.get("notes") or a.get("contact") or a.get("url"):
                with st.expander(f"Notes - {a.get('title','')}"):
                    if a.get("contact"): st.markdown(f"Contact: {a['contact']}")
                    if a.get("notes"):   st.markdown(a["notes"])
                    if a.get("url"):     st.markdown(f"[Open job]({a['url']})")
            new_s = st.selectbox("",STATUSES,index=STATUSES.index(a.get("status","Saved")) if a.get("status") in STATUSES else 0,key=f"s_{a.get('id',idx)}",label_visibility="collapsed")
            if new_s != a.get("status"):
                for item in DATA["applications"]:
                    if item.get("id")==a.get("id"): item["status"]=new_s
                save_data(DATA); st.rerun()
            st.markdown("<hr>",unsafe_allow_html=True)
    if apps:
        df = pd.DataFrame(apps)
        st.download_button("Download CSV", df.to_csv(index=False), "lets_get_hired.csv", "text/csv")

# ═══════════════════════════════════════════════════════════════════════════════
# CV EDITOR
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "CV Editor":
    st.markdown("## CV Editor")
    cv = DATA["cv"]
    col_e,col_p = st.columns(2)
    with col_e:
        cv["name"]        = st.text_input("Full name",    value=cv.get("name",""))
        cv["contact"]     = st.text_input("Contact",      value=cv.get("contact",""))
        cv["summary"]     = st.text_area("Summary",       value=cv.get("summary",""),   height=100)
        cv["skills"]      = st.text_area("Skills",        value=cv.get("skills",""),    height=70)
        st.markdown("**Experience 1**")
        cv["exp1_title"]  = st.text_input("Title",        value=cv.get("exp1_title",""),key="e1t")
        cv["exp1_dates"]  = st.text_input("Dates",        value=cv.get("exp1_dates",""),key="e1d")
        cv["exp1_bullets"]= st.text_area("Bullets",       value=cv.get("exp1_bullets",""),key="e1b",height=90)
        st.markdown("**Experience 2**")
        cv["exp2_title"]  = st.text_input("Title",        value=cv.get("exp2_title",""),key="e2t")
        cv["exp2_dates"]  = st.text_input("Dates",        value=cv.get("exp2_dates",""),key="e2d")
        cv["exp2_bullets"]= st.text_area("Bullets",       value=cv.get("exp2_bullets",""),key="e2b",height=70)
        cv["education"]   = st.text_area("Education",     value=cv.get("education",""), height=70)
        if st.button("Save CV"): DATA["cv"]=cv; save_data(DATA); st.success("CV saved!")
        plain = f"{cv.get('name','')}\n{cv.get('contact','')}\n\nSUMMARY\n{cv.get('summary','')}\n\nSKILLS\n{cv.get('skills','')}\n\nEXPERIENCE\n{cv.get('exp1_title','')}\n{cv.get('exp1_dates','')}\n{cv.get('exp1_bullets','')}\n\n{cv.get('exp2_title','')}\n{cv.get('exp2_dates','')}\n{cv.get('exp2_bullets','')}\n\nEDUCATION\n{cv.get('education','')}"
        st.download_button("Download CV (.txt)", plain, "devanshi_cv.txt", "text/plain")
    with col_p:
        st.markdown("**Preview**")
        def bl(t): return "".join(f"<div style='margin:1px 0'>{b}</div>" for b in t.split("\n") if b.strip())
        st.markdown(f"""<div style="background:white;border-radius:12px;padding:1.5rem;color:#1a1a1a;font-size:12px;line-height:1.6">
            <div style="font-size:18px;font-weight:700">{cv.get('name','')}</div>
            <div style="font-size:10px;color:#555;margin-bottom:10px">{cv.get('contact','')}</div>
            <div style="font-size:9px;font-weight:600;text-transform:uppercase;letter-spacing:1px;color:#7c3aed;border-bottom:1px solid #e9d5ff;margin:8px 0 5px">Summary</div>
            <p style="font-size:11px">{cv.get('summary','')}</p>
            <div style="font-size:9px;font-weight:600;text-transform:uppercase;letter-spacing:1px;color:#7c3aed;border-bottom:1px solid #e9d5ff;margin:8px 0 5px">Skills</div>
            <p style="font-size:11px">{cv.get('skills','')}</p>
            <div style="font-size:9px;font-weight:600;text-transform:uppercase;letter-spacing:1px;color:#7c3aed;border-bottom:1px solid #e9d5ff;margin:8px 0 5px">Experience</div>
            <div style="margin-bottom:8px"><div style="font-weight:600;font-size:11px">{cv.get('exp1_title','')}</div>
            <div style="font-size:10px;color:#888">{cv.get('exp1_dates','')}</div>
            <div style="font-size:11px">{bl(cv.get('exp1_bullets',''))}</div></div>
            <div style="margin-bottom:8px"><div style="font-weight:600;font-size:11px">{cv.get('exp2_title','')}</div>
            <div style="font-size:10px;color:#888">{cv.get('exp2_dates','')}</div>
            <div style="font-size:11px">{bl(cv.get('exp2_bullets',''))}</div></div>
            <div style="font-size:9px;font-weight:600;text-transform:uppercase;letter-spacing:1px;color:#7c3aed;border-bottom:1px solid #e9d5ff;margin:8px 0 5px">Education</div>
            <p style="font-size:11px">{cv.get('education','').replace(chr(10),'<br>')}</p>
        </div>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# INTERVIEW PREP
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Interview Prep":
    st.markdown("## Interview Prep")
    st.markdown("STAR answers tailored to your background - Devanshi.")
    tabs = st.tabs(["OpenAI","TikTok","Google","Meta","General T&S"])
    qa_sets = [
        [("Tell me about yourself","3+ years Trust & Safety at Meta via Covalen. Sole market owner across 4 markets. LLM evaluation, abuse detection, content policy. MSc Business Analytics. CSPO. Now seeking a permanent senior T&S role at an AI-first company like OpenAI."),
         ("Why OpenAI?","Your T&S team is building safety infrastructure for the most consequential AI of our time. My LLM evaluation and content policy background maps directly - I understand both how models fail AND how to scale enforcement operationally."),
         ("Describe a complex T&S case you owned","Use the coordinated account network investigation: SQL analysis to identify signal, cross-team escalation, policy recommendation, enforcement action, and measurable outcome."),
         ("What do you know about DSA and EU AI Act?","DSA requires transparency reports, risk assessments for VLOPs, and regulator responsiveness. EU AI Act classifies AI by risk level - GPAIs like ChatGPT fall under Article 51.")],
        [("Why TikTok?","TikTok has one of the most complex T&S environments globally. Short-form video at scale, multilingual markets, live content. My EMEA market ownership and LLM evaluation experience is directly applicable."),
         ("How do you analyse safety data at scale?","SQL for pattern detection, Python/Pandas for trend analysis, BI dashboards for reporting. At Meta I built dashboards tracking policy enforcement metrics across 4 markets."),
         ("Tell me about a policy you helped improve","Data-driven gap identification - proposed policy change - cross-functional review - A/B test - rollout. Always anchor on measurable outcome.")],
        [("Tell me about yourself - data analyst framing","I have spent 3 years using SQL and Python to identify coordinated account networks, abuse patterns, and policy violations at Meta. My work sits at the intersection of data analysis and integrity operations."),
         ("Walk me through a SQL analysis","Coordinated account detection: GROUP BY device fingerprint, COUNT of accounts, suspicious temporal clustering. Found networks of fake accounts. Escalated to policy team with full evidence package."),
         ("How do you measure T&S impact?","Leading: detection rate, false positive rate, escalation latency. Lagging: repeat violation rate, user harm reports. Built dashboards tracking all of these across 4 markets.")],
        [("You worked for Meta before - what would you do differently?","Focus more on proactive rather than reactive enforcement. Build better tooling for market owners. Invest in cross-market knowledge sharing."),
         ("How do you prioritise across multiple markets?","Risk-based scoring: severity x volume x regulatory exposure. Used a triage matrix to allocate effort across 4 markets efficiently.")],
        [("Why are you leaving / what happened?","My contract with Covalen concluded as Meta consolidated its vendor operations. I am now actively pursuing permanent senior roles in T&S and AI analysis."),
         ("Biggest strength?","Combining technical data skills with operational T&S judgment. I can write the SQL query AND translate findings into a policy recommendation AND present to stakeholders."),
         ("Where in 3 years?","Senior T&S Analyst or AI Policy Manager at a major platform, owning a product area end-to-end and contributing to AI governance frameworks.")],
    ]
    for tab,qa_list in zip(tabs,qa_sets):
        with tab:
            for q_text,a_text in qa_list:
                with st.expander(f"? {q_text}"):
                    st.markdown(a_text)
                    st.text_area("Your notes", key=f"note_{q_text[:15]}", height=70, placeholder="Add your own notes...")
    st.markdown("---")
    st.info("**STAR reminder**: Situation - Task - Action - Result. Always end with a number or measurable outcome.")
    st.markdown("#### Your unique selling points")
    st.markdown("""
- **Rare combo**: LLM evaluation + T&S operations + data analysis - very few people have all three
- **EMEA market ownership**: autonomous policy decisions for a region across 4 markets
- **EU AI Act awareness**: directly relevant for OpenAI, Google, TikTok Dublin right now
- **MSc Business Analytics**: signals data fluency beyond just operations
- **CSPO certified**: product thinking on top of analyst skills
- **Immediately available**: top of recruiter shortlists
    """)

# ═══════════════════════════════════════════════════════════════════════════════
# SALARY GUIDE
# ═══════════════════════════════════════════════════════════════════════════════

elif page == "Silicon Valley & AI News":
    st.markdown("## Silicon Valley & Global AI News")
    st.markdown("Stay informed on the trends shaping your job market - updated April 2026.")

    st.markdown("""<div class="tip-box">
        <div style="font-size:12px;font-weight:600;color:#93c5fd;margin-bottom:4px">Why this matters for your job search</div>
        <div style="font-size:13px;color:#dbeafe;line-height:1.5">
        EU AI Act full enforcement starts <strong>August 2, 2026</strong> - just 99 days away.
        Companies are urgently hiring AI governance and T&S analysts RIGHT NOW. Your timing is perfect.
        </div>
    </div>""", unsafe_allow_html=True)

    tag_f = st.selectbox("Filter by topic", ["All","AI","Policy","Jobs","Funding"])
    news = SV_NEWS
    if tag_f != "All": news = [n for n in news if n["tag"] == tag_f]

    for item in news:
        tag_colors = {"AI":"background:#1e3a5f;color:#93c5fd","Policy":"background:#4c1d95;color:#ddd6fe","Jobs":"background:#064e3b;color:#6ee7b7","Funding":"background:#78350f;color:#fde68a"}
        tag_style = tag_colors.get(item["tag"],"background:#2d1063;color:#a78bfa")
        st.markdown(f"""<div class="card">
            <div style="display:flex;align-items:flex-start;justify-content:space-between;gap:10px;flex-wrap:wrap">
                <div style="flex:1;min-width:200px">
                    <a href="{item['url']}" target="_blank" style="font-size:13px;font-weight:500;color:#a78bfa;text-decoration:none;line-height:1.4;display:block">{item['title']}</a>
                    <div style="font-size:11px;color:#7c3aed;margin-top:4px">{item['source']} - {item['date']}</div>
                    <div style="font-size:11px;color:#6ee7b7;margin-top:3px;font-style:italic">Why it matters: {item['relevance']}</div>
                </div>
                <span style="{tag_style};padding:2px 8px;border-radius:8px;font-size:10px;font-weight:600;white-space:nowrap">{item['tag']}</span>
            </div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### Stay updated - follow these sources daily")
    for label, url in [
        ("TechCrunch - AI news","https://techcrunch.com/category/artificial-intelligence/"),
        ("VentureBeat - AI & enterprise","https://venturebeat.com/category/ai"),
        ("Silicon Republic - Irish tech","https://www.siliconrepublic.com/machines"),
        ("The Register - Tech news","https://www.theregister.com/"),
        ("Wired - Tech culture & policy","https://www.wired.com/tag/artificial-intelligence/"),
        ("MIT Technology Review","https://www.technologyreview.com/topic/artificial-intelligence/"),
        ("EU AI Act tracker","https://artificialintelligenceact.eu/"),
        ("80,000 Hours - AI safety jobs","https://80000hours.org/jobs/"),
    ]:
        st.markdown(f"[{label}]({url})")



elif page == "All Dublin Companies":
    st.markdown("## All Companies in Dublin")
    st.markdown(f"**{len(ALL_COMPANIES)} companies** with Dublin offices - every sector covered.")

    col1, col2, col3 = st.columns(3)
    with col1: search_c = st.text_input("Search", placeholder="e.g. Google, Stripe, Fenergo...")
    with col2: cat_f2   = st.selectbox("Category", ["All","Big Tech","AI","Fintech","SaaS","Security","Platforms","Consulting","Finance","Gaming","Startup"])
    with col3: show_desc = st.checkbox("Show descriptions", value=True)

    cos = ALL_COMPANIES
    if search_c: cos = [c for c in cos if search_c.lower() in c[0].lower()]
    if cat_f2 != "All": cos = [c for c in cos if c[2] == cat_f2]

    st.caption(f"{len(cos)} companies shown")

    cat_colors2 = {
        "Big Tech":"#4c1d95","AI":"#1e3a5f","Fintech":"#064e3b",
        "SaaS":"#78350f","Security":"#7f1d1d","Platforms":"#1f2937",
        "Consulting":"#374151","Finance":"#1e3a5f","Gaming":"#064e3b","Startup":"#713f12",
    }

    # Group by category
    groups = {}
    for c in cos: groups.setdefault(c[2],[]).append(c)

    for cat, items in groups.items():
        color = cat_colors2.get(cat, "#2d1063")
        st.markdown(f"""<div style="font-size:12px;font-weight:600;text-transform:uppercase;letter-spacing:0.8px;color:#a78bfa;margin:14px 0 8px;padding-left:4px;border-left:3px solid {color}">
            {cat} ({len(items)} companies)</div>""", unsafe_allow_html=True)
        cols2 = st.columns(3)
        for i,(name,abbr,cat2,desc,url) in enumerate(items):
            with cols2[i%3]:
                bg = cat_colors2.get(cat2,"#2d1063")
                desc_html = f'<div style="font-size:10px;color:#9ca3af;margin:2px 0 5px;line-height:1.3">{desc}</div>' if show_desc else '<div style="margin:5px 0"></div>'
                st.markdown(f"""<div class="agency-card">
                    <div style="display:flex;align-items:center;gap:8px;margin-bottom:3px">
                        <div style="width:26px;height:26px;border-radius:50%;background:{bg};color:#e9d5ff;display:flex;align-items:center;justify-content:center;font-size:9px;font-weight:600;flex-shrink:0">{abbr}</div>
                        <span style="font-size:13px;font-weight:500;color:#ddd6fe">{name}</span>
                    </div>
                    {desc_html}
                    <a href="{url}" target="_blank" style="font-size:11px;background:#2d1063;color:#a78bfa;padding:2px 10px;border-radius:6px;text-decoration:none">Jobs</a>
                </div>""", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)



elif page == "Weekly Job Plan":
    st.markdown("## Weekly Job Search Plan")
    st.markdown("Follow this plan every week to maximise your chances of getting hired within a month.")

    st.markdown("""<div class="quote-box">
        <div style="font-size:14px;color:#ddd6fe;line-height:1.6">Job searching is a full-time job. The candidates who get hired fastest are not the ones with the best CV - they are the ones who are most systematic, most persistent, and most visible. This plan makes you that person.</div>
    </div>""", unsafe_allow_html=True)

    st.markdown("### Daily routine - do these every single day")
    for task in [
        ("Check LinkedIn jobs - Trust & Safety + AI Analyst + Dublin - filter by past 24 hours", "https://www.linkedin.com/jobs/search/?keywords=trust+safety+analyst&location=Dublin&f_TPR=r86400"),
        ("Check Indeed Ireland - analyst + Dublin - sorted by date", "https://ie.indeed.com/jobs?q=trust+safety+analyst+OR+AI+analyst+OR+business+analyst&l=Dublin&fromage=1&sort=date"),
        ("Read Silicon Republic - 5 minutes of Irish tech news", "https://www.siliconrepublic.com"),
        ("Send at least 2 applications", None),
        ("Check email for recruiter replies and follow up any outstanding applications over 5 days old", None),
    ]:
        label, url = task
        if url:
            st.markdown(f"- [ ] [{label}]({url})")
        else:
            st.markdown(f"- [ ] {label}")

    st.markdown("---")
    st.markdown("### Week 1 - Foundation (do these first)")
    for task in [
        ("Update LinkedIn headline to: Trust & Safety AI Analyst | LLM Evaluation | Immediately Available | Dublin", None),
        ("Set LinkedIn to Open to Work (visible to recruiters only)", None),
        ("Register with CPL Recruitment and send CV", "https://www.cpl.com/jobs"),
        ("Register with Morgan McKinley and send CV", "https://www.morganmckinley.com/ie/jobs"),
        ("Register with Sigmar Recruitment and send CV", "https://www.sigmarrecruitment.com/"),
        ("Register with Mason Alexander and send CV", "https://www.masonalexander.ie/jobs"),
        ("Set up daily email alert on LinkedIn for Trust & Safety + Dublin", "https://www.linkedin.com/jobs/search/?keywords=trust+safety+analyst&location=Dublin"),
        ("Set up daily email alert on Indeed IE for analyst + Dublin", "https://ie.indeed.com/jobs?q=analyst&l=Dublin&fromage=1"),
        ("Set up alert on IrishJobs", "https://www.irishjobs.ie/Jobs/analyst/in-Dublin"),
        ("Apply to OpenAI Trust & Safety Operations Analyst - TOP PRIORITY", "https://openai.com/careers/trust-and-safety-operations-analyst-2/"),
        ("Apply to TikTok Reporting & Insights Analyst - Youth Safety", "https://careers.tiktok.com/position?location=CT_211&query=analyst"),
        ("Apply to Meta GRO Intelligence Analyst", "https://www.metacareers.com/jobs?offices=Dublin&q=trust+integrity"),
        ("Tailor CV summary for T&S roles - lead with LLM evaluation", None),
    ]:
        label, url = task
        if url:
            st.markdown(f"- [ ] [{label}]({url})")
        else:
            st.markdown(f"- [ ] {label}")

    st.markdown("---")
    st.markdown("### Week 2 - Expand outreach")
    for task in [
        ("Register with Hays, Archer Recruitment, Solas IT and IT Search", None),
        ("Apply to Google Engineering Analyst AI Safety", "https://careers.google.com/jobs/results/99432678838674118-engineering-analyst/"),
        ("Apply to Accenture Trust & Safety Team Lead", "https://www.accenture.com/ie-en/careers/jobsearch?jk=trust+safety&cl=Dublin"),
        ("Apply to EY Business Analyst - Financial Services Consulting", "https://www.ey.com/en_ie/careers"),
        ("Apply to Deloitte Technology Business Analyst - Big Data", "https://apply.deloitte.com/careers/SearchJobs/analyst?3_56_3=5440"),
        ("Apply to AI Governance Analyst - Irish Life", "https://www.irishjobs.ie/Jobs/analyst/in-Dublin"),
        ("Connect with 5 hiring managers on LinkedIn - T&S and AI teams", None),
        ("Follow up all Week 1 applications that have not replied", None),
        ("Create Wellfound profile and set job preferences", "https://wellfound.com/jobs?location=dublin"),
        ("Create Otta profile and set preferences", "https://otta.com/jobs/search?location=Dublin"),
    ]:
        label, url = task
        if url:
            st.markdown(f"- [ ] [{label}]({url})")
        else:
            st.markdown(f"- [ ] {label}")

    st.markdown("---")
    st.markdown("### Week 3 - Go deeper")
    for task in [
        ("Apply to Whatnot Senior Manager Trust & Safety", "https://www.whatnot.com/careers"),
        ("Apply to LILT Product Owner - Localization AI", "https://lilt.com/careers"),
        ("Apply to Intercom Data Analyst", "https://www.intercom.com/careers"),
        ("Apply to Fenergo Business Analyst", "https://www.fenergo.com/company/careers/"),
        ("Apply to Salesforce Policy Operations Analyst - Slack", "https://careers.salesforce.com/en/jobs/?search=analyst&location=Dublin"),
        ("Apply to Bank of Ireland Product Owner - Digital Banking", "https://careers.bankofireland.com"),
        ("Follow up Week 1 and Week 2 applications - send a brief check-in email", None),
        ("Ask each recruitment agency if they have any new matches", None),
        ("Research EU AI Act - prepare 3 talking points for interviews", "https://artificialintelligenceact.eu/"),
        ("Do 2 mock STAR interviews - record yourself on phone", None),
    ]:
        label, url = task
        if url:
            st.markdown(f"- [ ] [{label}]({url})")
        else:
            st.markdown(f"- [ ] {label}")

    st.markdown("---")
    st.markdown("### Week 4 - Close and convert")
    for task in [
        ("Follow up every single application you sent in Weeks 1-3", None),
        ("Ask every agency contact: do you have anything new this week?", None),
        ("Apply to 10 new roles from Built In Dublin and startup boards", "https://builtindublin.ie/jobs"),
        ("Apply via any new roles posted on Silicon Republic Jobs", "https://www.siliconrepublic.com/jobs"),
        ("Post on LinkedIn about your job search - your network can help", None),
        ("Ask former Meta/Covalen colleagues for LinkedIn recommendations", None),
        ("Review your tracker - identify any companies that have not responded - reach out directly", None),
        ("Prepare references - have 2-3 ready to share immediately if asked", None),
    ]:
        label, url = task
        if url:
            st.markdown(f"- [ ] [{label}]({url})")
        else:
            st.markdown(f"- [ ] {label}")

    st.markdown("---")
    st.markdown("""<div class="fun-fact">
        <div style="font-size:12px;font-weight:600;color:#6ee7b7;margin-bottom:4px">The numbers game</div>
        <div style="font-size:13px;color:#d1fae5;line-height:1.5">
        In Dublin tech, a typical job search pipeline looks like this:<br>
        40 applications - 8 recruiter calls - 4 first interviews - 2 second interviews - 1 offer.<br>
        If you follow this 4-week plan, you can hit those numbers. Keep going.
        </div>
    </div>""", unsafe_allow_html=True)


elif page == "Salary Guide":
    st.markdown("## Dublin Salary Guide 2026")
    st.markdown("Salary ranges for your target roles in Dublin tech companies.")

    st.markdown("""<div class="tip-box">
        <div style="font-size:12px;font-weight:600;color:#93c5fd;margin-bottom:4px">Devanshi's target range</div>
        <div style="font-size:13px;color:#dbeafe">Based on your 3+ years T&S experience, MSc and CSPO cert, you should be targeting EUR55,000-80,000 depending on seniority and company size. Do not undersell yourself.</div>
    </div>""", unsafe_allow_html=True)

    salary_data = [
        ("Trust & Safety Analyst",         "Mid",    "45-60k",  "TikTok, Meta, Accenture"),
        ("Trust & Safety Analyst",         "Senior", "60-80k",  "OpenAI, Meta, Google"),
        ("Trust & Safety Manager",         "Senior", "75-100k", "OpenAI, TikTok, Google"),
        ("AI Analyst / LLM Evaluator",     "Mid",    "50-70k",  "Google, Scale AI, Anthropic"),
        ("AI Analyst / LLM Evaluator",     "Senior", "65-90k",  "Google, OpenAI, Anthropic"),
        ("AI Governance Analyst",          "Mid",    "55-75k",  "Irish Life, AIB, Big Tech"),
        ("Data Analyst",                   "Mid",    "45-60k",  "Stripe, HubSpot, Meta"),
        ("Data Analyst",                   "Senior", "60-80k",  "Google, Stripe, TikTok"),
        ("Business Analyst",               "Mid",    "45-65k",  "EY, Deloitte, Accenture"),
        ("Business Analyst",               "Senior", "60-80k",  "Big Tech, Consulting firms"),
        ("Product Owner",                  "Mid",    "55-75k",  "Bank of Ireland, HubSpot"),
        ("Product Owner",                  "Senior", "70-95k",  "Anthropic, Big Tech, Fintech"),
        ("Policy Analyst",                 "Mid",    "50-70k",  "TikTok, Meta, Google"),
        ("Risk & Compliance Analyst",      "Mid",    "50-70k",  "Revolut, AIB, JP Morgan"),
    ]

    df_sal = pd.DataFrame(salary_data, columns=["Role","Level","Salary (EUR)","Example companies"])
    st.dataframe(df_sal, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("#### Salary benchmarking resources")
    for label,url in [
        ("Sigmar 2026 Salary Guide","https://www.sigmarrecruitment.com/salary-guide/"),
        ("Hays Ireland 2026 Salary Guide","https://www.hays.ie/salary-guide"),
        ("Mason Alexander 2026 Tech Salary Guide","https://www.masonalexander.ie/salary-guide"),
        ("IT Search 2026 Salary Guide","https://itsearch.ie/salary-guide/"),
        ("Glassdoor - Trust & Safety salaries Dublin","https://www.glassdoor.ie/Salaries/dublin-trust-and-safety-analyst-salary-SRCH_IL.0,6_IM1078_KO7,31.htm"),
        ("LinkedIn Salary Insights","https://www.linkedin.com/salary/"),
    ]:
        st.markdown(f"[{label}]({url})")

    st.markdown("---")
    st.markdown("#### Negotiation tips for your next offer")
    st.markdown("""
- **Always negotiate.** 85% of employers expect it. First offer is rarely the best offer.
- **Anchor high.** If they ask your expectation, say EUR5-10k above your minimum.
- **Use competing offers.** Even agency interest counts as leverage.
- **Ask about the full package**: pension (5-10% is standard), health insurance, bonus (5-15%), training budget, hybrid flexibility.
- **Your LLM evaluation background commands a premium** right now due to EU AI Act demand. Use it.
    """)
