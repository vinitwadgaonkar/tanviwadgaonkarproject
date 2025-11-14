# Vercel Deployment Instructions

## Quick Deploy

Your project is ready for Vercel deployment! Here are the steps:

### Option 1: Using Vercel CLI (Recommended)

1. Install Vercel CLI (if not already installed):
```bash
npm i -g vercel
```

2. Deploy:
```bash
vercel --prod
```

### Option 2: Using Vercel Dashboard

1. Go to [vercel.com](https://vercel.com)
2. Click "New Project"
3. Import your Git repository (or drag & drop the project folder)
4. Vercel will auto-detect the Python/FastAPI setup
5. Click "Deploy"

### Option 3: Using npx (No installation needed)

```bash
npx vercel --prod
```

## Project Structure

- `api/index.py` - FastAPI application (entry point)
- `requirements.txt` - Python dependencies
- `vercel.json` - Vercel configuration
- `models/` - ML models (will be included in deployment)

## Important Notes

- The API will be available at: `https://your-project.vercel.app`
- All 4 endpoints will work:
  - `/crop_recommend`
  - `/yield_predict`
  - `/fertilizer_recommend`
  - `/weather_forecast`
- API docs: `https://your-project.vercel.app/docs`

## After Deployment

Update `index.html` to use your Vercel URL:
```javascript
const API_URL = 'https://your-project.vercel.app';
```

