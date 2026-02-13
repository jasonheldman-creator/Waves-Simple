# WAVES Intelligence™

This repository contains two applications:

## 1. Marketing Site (Next.js)
**Location:** `/site` directory  
**Technology:** Next.js 16, React 19, TypeScript, Tailwind CSS  
**Deployment:** Vercel

### Quick Start
```bash
cd site
npm install
npm run dev
```

### Deployment to Vercel
The marketing site is deployed to Vercel. **Important**: The site lives in the `/site` subdirectory.

#### Fixing 404 Errors on Vercel
If you're seeing a 404 NOT_FOUND error, see [DEPLOYMENT_FIX.md](./DEPLOYMENT_FIX.md) for configuration instructions.

**Key Requirements:**
- Set **Root Directory** to `site` in Vercel Dashboard → Settings → General
- Framework should auto-detect as Next.js
- No custom build commands needed

Full deployment guide: [VERCEL_SETUP.md](./VERCEL_SETUP.md)

## 2. Analytics Application (Streamlit)
**Location:** Root directory (`.py` files)  
**Technology:** Python, Streamlit  
**Deployment:** Streamlit Cloud

### Quick Start
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Repository Structure
```
Waves-Simple/
├── site/                      # Next.js marketing website
│   ├── src/
│   │   ├── app/              # Next.js App Router pages
│   │   ├── components/       # React components
│   │   └── ...
│   ├── package.json
│   ├── next.config.ts
│   └── README.md
│
├── app.py                     # Streamlit analytics app
├── requirements.txt           # Python dependencies
├── vercel.json               # Vercel configuration (redirects only)
├── DEPLOYMENT_FIX.md         # Fix for Vercel 404 errors
├── VERCEL_SETUP.md           # Complete Vercel deployment guide
└── ...
```

## Documentation
- **Marketing Site Deployment**: [VERCEL_SETUP.md](./VERCEL_SETUP.md)
- **Fixing Vercel 404 Errors**: [DEPLOYMENT_FIX.md](./DEPLOYMENT_FIX.md)
- **Marketing Site Details**: [site/README.md](./site/README.md)

