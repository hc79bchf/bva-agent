# BVA Agent - Deployment Guide

## Vercel Deployment (Static Wireframe)

### Quick Deploy

The wireframe is now configured for Vercel deployment with proper routing.

**Files:**
- `index.html` - Entry point that redirects to the wireframe
- `bva-agent-wireframe.html` - Main wireframe application
- `vercel.json` - Vercel configuration for routing

### Deploy Steps

1. **Push changes to Git:**
   ```bash
   git push origin main
   ```

2. **Vercel will automatically redeploy** if you have it connected to your repository.

   Or manually deploy:
   ```bash
   vercel --prod
   ```

3. **Access your deployment:**
   - Your site will be available at: `https://your-project.vercel.app`
   - The root path `/` will show the BVA Agent wireframe

### Troubleshooting

**404 Error:**
- Make sure you've pushed the latest changes including `index.html` and `vercel.json`
- Redeploy the project in Vercel dashboard
- Check that the deployment includes all HTML files

**Static Files Not Loading:**
- Verify `vercel.json` is in the root directory
- Ensure all routes are configured correctly

### Current Configuration

The `vercel.json` configures:
- All HTML files as static assets
- Root path `/` routes to `index.html`
- All other paths serve their respective files

### Local Testing

Before deploying, test locally:

```bash
# Install Vercel CLI if needed
npm i -g vercel

# Run local dev server
vercel dev
```

Then open `http://localhost:3000` in your browser.

## Future: FastAPI Backend Deployment

When ready to deploy the FastAPI backend, you'll need:

1. **Update `vercel.json` for serverless functions:**
   ```json
   {
     "builds": [
       { "src": "api/main.py", "use": "@vercel/python" }
     ],
     "routes": [
       { "src": "/api/(.*)", "dest": "api/main.py" }
     ]
   }
   ```

2. **Add `requirements.txt`** in root or `api/` directory

3. **Move Python code** to `api/` directory

4. **Configure environment variables** in Vercel dashboard for:
   - Database connection strings
   - API keys
   - Other secrets
