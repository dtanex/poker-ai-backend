# 🚀 DEPLOYMENT INSTRUCTIONS

Your poker AI is ready to deploy! Follow these steps exactly:

## Step 1: Create GitHub Repo (2 minutes)

1. Go to: https://github.com/new
2. Repository name: `poker-ai-backend`
3. Description: `CFR-based GTO poker AI with FastAPI backend`
4. Make it **PUBLIC** ✅
5. **DON'T** initialize with README (we already have one)
6. Click **"Create repository"**

## Step 2: Push Code to GitHub (1 minute)

Copy these commands and run them in your terminal:

```bash
cd /Users/davidtanchin/Desktop/poker-ai
git remote add origin https://github.com/YOUR_USERNAME/poker-ai-backend.git
git branch -M main
git push -u origin main
```

Replace `YOUR_USERNAME` with your actual GitHub username!

## Step 3: Deploy to Render (5 minutes)

1. Go to: https://dashboard.render.com
2. Sign in with GitHub if you haven't already
3. Click **"New +" → "Web Service"**
4. Click **"Connect a repository"**
5. Find and select `poker-ai-backend`
6. Configure the service:

   **Basic Settings:**
   - Name: `poker-ai-api` (or whatever you want)
   - Region: `Oregon (US West)` (or closest to you)
   - Branch: `main`
   - Root Directory: (leave blank)

   **Build & Deploy:**
   - Runtime: `Python 3`
   - Build Command:
     ```
     pip install -r requirements.txt && python cfr_trainer.py
     ```
   - Start Command:
     ```
     uvicorn fastapi_backend:app --host 0.0.0.0 --port $PORT
     ```

   **Instance Type:**
   - Select: `Free` ($0/month)

7. Click **"Create Web Service"**

## Step 4: Wait for Deploy (3-5 minutes)

Render will:
- ✅ Clone your repo
- ✅ Install dependencies
- ✅ Train the CFR model (10k iterations)
- ✅ Start FastAPI server

Watch the logs in Render dashboard. You'll see:
```
🎰 CFR POKER STRATEGY TRAINER
🏋️ Training preflop strategies...
✅ Preflop training complete
🏋️ Training postflop strategies...
✅ Postflop training complete
💾 Strategies saved to cfr_strategies.pkl
🎉 TRAINING COMPLETE!
```

Then:
```
✅ AI player loaded successfully
🚀 Starting Poker AI API server...
INFO: Uvicorn running on http://0.0.0.0:10000
```

## Step 5: Test Your Live API (1 minute)

Once deploy succeeds, Render gives you a URL like:
`https://poker-ai-api.onrender.com`

Test it:

```bash
# Health check
curl https://YOUR-APP-URL.onrender.com/health

# Should return:
# {"status":"healthy","ai_loaded":true,"version":"1.0.0"}

# Test with pocket aces
curl -X POST https://YOUR-APP-URL.onrender.com/strategy \
  -H "Content-Type: application/json" \
  -d '{"stage":"preflop","hole_cards":["As","Ah"],"board":[]}'

# Should return GTO strategy with 99.9% raise
```

## Step 6: Save Your API URL

Copy your Render URL and save it. You'll need it to connect your Next.js frontend!

Example: `https://poker-ai-api.onrender.com`

---

## 🎉 YOU'RE LIVE!

Your poker AI is now deployed and accessible from anywhere!

**API Endpoints:**
- `GET /health` - Check if AI is loaded
- `GET /docs` - Interactive API documentation
- `POST /strategy` - Get GTO strategy
- `POST /action` - Get recommended action
- `POST /coaching` - Get natural language explanation

**Next Steps:**
1. Test all endpoints in the Render logs
2. Copy your API URL
3. Connect it to your Next.js frontend at `localhost:3000`
4. Build the integration in your analyze page

---

## 🔐 Adding Claude API Later (When Ready)

When you want to add Claude for natural language coaching:

1. Go to Render Dashboard → Your Service → Environment
2. Add environment variable:
   - Key: `ANTHROPIC_API_KEY`
   - Value: `your-api-key-here`
3. Save and redeploy

The code will automatically pick it up from `os.getenv("ANTHROPIC_API_KEY")`

**NO API KEY EVER GOES IN THE CODE!**

---

## 💰 Costs

- **Render Free Tier**: $0/month
- **Goes to sleep after 15 min of inactivity**
- **Cold start takes ~30 seconds** (first request after sleep)
- **Upgrade to $7/month** for always-on if needed

---

## 🚨 Troubleshooting

**Build fails?**
- Check Render logs for Python errors
- Make sure requirements.txt is correct
- Verify Python 3.9+ is being used

**AI not loading?**
- Check if cfr_trainer.py ran successfully in build logs
- Make sure cfr_strategies.pkl was created
- Look for "✅ AI player loaded successfully" in logs

**Port errors?**
- Make sure start command uses `$PORT` variable
- Render assigns the port automatically

---

## 📞 Need Help?

Check the logs in Render dashboard - they show everything!

Now go create that GitHub repo and deploy! 🚀
