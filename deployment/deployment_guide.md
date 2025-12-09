# Deployment Guide for RAG Chatbot

## 📦 Deployment Options

### Option 1: Streamlit Cloud (Recommended - Free & Easy)

#### Prerequisites
- GitHub account
- Google API key
- Your code pushed to a GitHub repository

#### Steps

1. **Prepare Your Repository**
   ```bash
   git init
   git add .
   git commit -m "Initial commit - RAG Chatbot"
   git branch -M main
   git remote add origin <your-github-repo-url>
   git push -u origin main
   ```

2. **Visit Streamlit Cloud**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Sign in with GitHub
   - Click "New app"

3. **Configure Deployment**
   - Repository: Select your repo
   - Branch: main
   - Main file path: `app.py`
   - App URL: Choose your custom URL

4. **Add Secrets**
   - Click "Advanced settings"
   - Go to "Secrets" section
   - Add your API key:
     ```toml
     GOOGLE_API_KEY = "your_api_key_here"
     ```

5. **Deploy**
   - Click "Deploy!"
   - Wait 2-3 minutes for deployment
   - Your app will be live at: `https://your-app-name.streamlit.app`

#### Important Notes
- Free tier limits: 1GB RAM, limited CPU
- App sleeps after inactivity (wakes up on first request)
- Public by default (can be made private)

---

### Option 2: Hugging Face Spaces (Free & GPU Available)

#### Steps

1. **Create a Space**
   - Go to [huggingface.co/spaces](https://huggingface.co/spaces)
   - Click "Create new Space"
   - Choose "Streamlit" as SDK
   - Name your space

2. **Upload Files**
   ```bash
   git clone https://huggingface.co/spaces/<your-username>/<space-name>
   cd <space-name>
   
   # Copy your files
   cp -r /path/to/your/project/* .
   
   # Push to HF
   git add .
   git commit -m "Deploy RAG Chatbot"
   git push
   ```

3. **Add Secrets**
   - Go to Space settings
   - Add secret: `GOOGLE_API_KEY`
   - Paste your API key

4. **Access Your App**
   - URL: `https://huggingface.co/spaces/<username>/<space-name>`

#### Benefits
- Free GPU access (for paid accounts)
- 16GB RAM on free tier
- Persistent storage available

---

### Option 3: Google Cloud Run (Scalable)

#### Prerequisites
- Google Cloud account
- gcloud CLI installed
- Docker installed

#### Steps

1. **Create Dockerfile**
   (Already provided in deployment/Dockerfile)

2. **Build and Push**
   ```bash
   # Set project
   gcloud config set project YOUR_PROJECT_ID
   
   # Build image
   docker build -t gcr.io/YOUR_PROJECT_ID/rag-chatbot .
   
   # Push to Container Registry
   docker push gcr.io/YOUR_PROJECT_ID/rag-chatbot
   ```

3. **Deploy to Cloud Run**
   ```bash
   gcloud run deploy rag-chatbot \
     --image gcr.io/YOUR_PROJECT_ID/rag-chatbot \
     --platform managed \
     --region us-central1 \
     --allow-unauthenticated \
     --set-env-vars GOOGLE_API_KEY=your_key_here
   ```

4. **Get URL**
   - Cloud Run will provide a URL
   - Example: `https://rag-chatbot-xxx.run.app`

---

### Option 4: Heroku (Easy but Paid)

#### Steps

1. **Create Heroku App**
   ```bash
   heroku login
   heroku create your-rag-chatbot
   ```

2. **Add Buildpack**
   ```bash
   heroku buildpacks:add heroku/python
   ```

3. **Set Environment Variables**
   ```bash
   heroku config:set GOOGLE_API_KEY=your_key_here
   ```

4. **Deploy**
   ```bash
   git push heroku main
   ```

5. **Open App**
   ```bash
   heroku open
   ```

---

### Option 5: Local with ngrok (Temporary Sharing)

#### For Quick Testing/Demo

1. **Install ngrok**
   - Download from [ngrok.com](https://ngrok.com)
   - Sign up for free account

2. **Run Your App**
   ```bash
   streamlit run app.py
   ```

3. **Expose with ngrok**
   ```bash
   ngrok http 8501
   ```

4. **Share URL**
   - ngrok provides a public URL
   - Example: `https://xxxx-xx-xx-xx-xx.ngrok.io`
   - Valid for 2 hours on free tier

---

## 🔧 Pre-Deployment Checklist

- [ ] `.env` file is in `.gitignore` (never commit API keys!)
- [ ] `requirements.txt` is up to date
- [ ] Sample documents are included (at least 5)
- [ ] Test the app locally first
- [ ] README.md is complete
- [ ] Collection is populated with documents
- [ ] API key is valid and has sufficient quota

---

## 📊 Post-Deployment Testing

After deployment, test:

1. **Document Upload**
   - Upload a new document
   - Verify it processes correctly

2. **Search Functionality**
   - Ask 5 different questions
   - Check if answers are relevant

3. **Performance**
   - Measure response times
   - Should be < 5 seconds

4. **Error Handling**
   - Ask irrelevant questions
   - Verify graceful error messages

---

## 🐛 Common Deployment Issues

### Issue: "Module not found"
**Solution**: Ensure all dependencies are in `requirements.txt`

### Issue: "Out of memory"
**Solution**: 
- Reduce `CHUNK_SIZE` in config
- Use smaller embedding model
- Deploy to platform with more RAM

### Issue: "API rate limit exceeded"
**Solution**:
- Add rate limiting in code
- Use API key with higher quota
- Implement caching

### Issue: "ChromaDB not persisting"
**Solution**:
- Check if persistent storage is enabled
- Verify write permissions
- Use external database for production

---

## 🎥 Demo Video Guidelines

Create a 3-minute demo showing:

1. **Introduction (30s)**
   - Project overview
   - Technologies used

2. **Document Upload (60s)**
   - Upload documents
   - Show processing

3. **Q&A Demo (60s)**
   - Ask 3-4 questions
   - Show sources and confidence

4. **Conclusion (30s)**
   - Summary
   - Future improvements

**Tools for Recording**:
- OBS Studio (free)
- Loom (easy sharing)
- Screen recording built into Windows/Mac

---

## 📱 Monitoring & Maintenance

### Track These Metrics:
- Response times
- Error rates
- API usage/costs
- User queries (for improvement)

### Regular Maintenance:
- Update dependencies monthly
- Review and expand document collection
- Retrain/update embeddings quarterly
- Monitor API costs

---

## 🚀 Scaling Considerations

When moving to production:

1. **Use Vector Database Service**
   - Pinecone
   - Weaviate
   - Milvus

2. **Implement Caching**
   - Cache frequent queries
   - Redis for session storage

3. **Add Authentication**
   - User management
   - Rate limiting per user

4. **Monitoring**
   - Application Performance Monitoring (APM)
   - Log aggregation
   - Alert systems

---

## 📞 Support

For deployment issues:
- Check Streamlit Community Forum
- GitHub Issues
- Course Slack: #deployment-help

---

**Note**: For student projects, Streamlit Cloud or Hugging Face Spaces are recommended for their free tiers and ease of use.
