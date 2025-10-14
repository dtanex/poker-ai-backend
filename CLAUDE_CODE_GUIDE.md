# 🤖 Using Claude Code with Your Poker AI

## What is Claude Code?

Claude Code is a command-line tool that lets Claude help you with coding tasks directly from your terminal. It's like having an AI pair programmer.

## Quick Start with Claude Code

### 1. Make sure Claude Code is installed
```bash
# Check if installed
claude-code --version

# If not installed, follow instructions at:
# https://docs.claude.com/en/docs/claude-code
```

### 2. Navigate to your project directory
```bash
cd /path/to/your/poker-ai-project
```

### 3. Place all the files in the same directory

Your directory should look like:
```
poker-ai/
├── aiplayer.py
├── cfr_trainer.py
├── fastapi_backend.py
├── test_complete.py
├── requirements.txt
├── CRITICAL_REVIEW.md
├── README.md
└── (cfr_strategies.pkl will be generated)
```

## How to Communicate Files to Claude Code

### Method 1: Ask Claude Code to read them (RECOMMENDED)

```bash
# Start Claude Code in your project directory
claude-code

# Then in the chat:
```

**Example conversation:**
```
You: Read all the Python files in this directory and tell me what the poker AI does

Claude: [reads aiplayer.py, cfr_trainer.py, etc. and explains]

You: Now help me deploy this to Render

Claude: [provides step-by-step deployment instructions]
```

### Method 2: Reference files explicitly

```bash
claude-code "Read aiplayer.py and fastapi_backend.py. How do I connect these to my Next.js frontend?"
```

### Method 3: Use file attachments (if supported)

Some versions of Claude Code let you attach files:
```bash
claude-code --file aiplayer.py --file fastapi_backend.py "Review this code"
```

## Common Tasks with Claude Code

### Task 1: Train the AI and test it

```bash
claude-code "Help me run the training pipeline:
1. Run cfr_trainer.py to generate strategies
2. Run test_complete.py to verify everything works
3. Show me the results"
```

Claude will:
- Run `python cfr_trainer.py`
- Wait for it to complete
- Run `python test_complete.py`
- Show you the test results
- Explain any issues

### Task 2: Deploy to Render

```bash
claude-code "I want to deploy this FastAPI backend to Render. Walk me through it step by step."
```

Claude will:
- Help you create a Render account
- Guide you through the deployment
- Help you configure environment variables
- Test the deployed API

### Task 3: Integrate with Next.js frontend

```bash
claude-code "My Next.js app is at /path/to/supalaunch. 
Help me create an API client to connect to the FastAPI backend.
The backend is at https://my-poker-api.onrender.com"
```

Claude will:
- Create an API client file
- Show you how to call the endpoints
- Help you integrate with your UI

### Task 4: Add Claude API coaching

```bash
claude-code "Read fastapi_backend.py. 
The /coaching endpoint needs Claude API integration.
Help me add natural language explanations using the Anthropic SDK."
```

Claude will:
- Add `anthropic` to requirements.txt
- Implement the coaching endpoint
- Show you how to set the API key
- Test the integration

### Task 5: Debug issues

```bash
claude-code "I'm getting this error when running test_complete.py:
[paste error message]

Debug it for me."
```

Claude will:
- Analyze the error
- Find the root cause
- Fix the code
- Verify the fix works

## Pro Tips for Claude Code

### 1. Be specific about context
```bash
# ❌ Bad
claude-code "fix the bug"

# ✅ Good
claude-code "There's a KeyError in aiplayer.py line 45 when I call get_strategy() with a new bucket. Fix it."
```

### 2. Ask for explanations
```bash
claude-code "Explain how the CFR training algorithm works in cfr_trainer.py. 
Why do we use regret matching?"
```

### 3. Request documentation
```bash
claude-code "Add docstrings to all functions in fastapi_backend.py 
so other developers can understand the API."
```

### 4. Get deployment help
```bash
claude-code "I need to deploy this to production. 
1. Create a Dockerfile
2. Add environment variable management
3. Set up logging
4. Add rate limiting"
```

### 5. Optimize performance
```bash
claude-code "The CFR training is too slow. 
Profile cfr_trainer.py and optimize the bottlenecks."
```

## Example: Complete Deployment Workflow

Here's a full conversation flow for deploying your poker AI:

```bash
# Step 1: Review code
claude-code "Review all Python files and tell me if there are any bugs or issues before deployment"

# Step 2: Train the AI
claude-code "Run python cfr_trainer.py and show me the output"

# Step 3: Test everything
claude-code "Run python test_complete.py. If any tests fail, fix them."

# Step 4: Prepare for deployment
claude-code "I want to deploy to Render. 
1. Check if requirements.txt has all dependencies
2. Create a start script
3. Explain the deployment steps"

# Step 5: Deploy
claude-code "Walk me through deploying to Render step by step. 
Wait for my confirmation after each step."

# Step 6: Test deployed API
claude-code "Test the deployed API at https://my-api.onrender.com/health
If it works, show me how to call the /strategy endpoint."

# Step 7: Connect frontend
claude-code "Now help me integrate this with my Next.js app in /path/to/supalaunch"
```

## What Claude Code CAN Help With

✅ Running Python scripts  
✅ Debugging errors  
✅ Explaining code  
✅ Deploying to cloud services  
✅ Creating config files (Dockerfile, etc.)  
✅ Writing tests  
✅ Optimizing performance  
✅ Adding features  
✅ Integrating with other services  

## What Claude Code CANNOT Do

❌ Access external URLs directly (but can help you write code to do it)  
❌ Run for hours (sessions have time limits)  
❌ Access your git credentials (you'll need to push manually)  

## Alternative: Use Regular Claude + Copy/Paste

If you don't have Claude Code, you can still use regular Claude:

1. **Paste the files into chat:**
```
Here's my aiplayer.py:
[paste full file]

Here's my fastapi_backend.py:
[paste full file]

Help me deploy this to Render.
```

2. **Claude will analyze and help**

3. **Copy the commands/code Claude provides and run them yourself**

It's slower than Claude Code but works the same way.

## Getting Help

If you're stuck:

1. **Check the docs:** https://docs.claude.com/en/docs/claude-code
2. **Ask Claude:** "I'm confused about how to use Claude Code for my poker AI project. Explain it step by step."
3. **Start simple:** Just try `claude-code "Read aiplayer.py and explain what it does"`

## Summary: Your Workflow

```bash
# 1. Navigate to project
cd ~/poker-ai

# 2. Start Claude Code
claude-code

# 3. In the Claude Code chat:
"Read all the Python files. I want to:
1. Train the CFR strategies
2. Test everything
3. Deploy to Render
4. Connect to my Next.js frontend

Walk me through it step by step."

# Claude will guide you through everything!
```

That's it! Claude Code makes this whole process way easier than doing it manually.
