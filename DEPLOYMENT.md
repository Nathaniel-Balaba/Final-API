# Deployment Guide

This guide covers different ways to deploy the Rice Leaf Disease Classification API.

## Quick Start

### 1. Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Train the model (if not already done)
python train.py

# Start the API
python app.py
```

The API will be available at `http://localhost:5000`

### 2. Docker Deployment

#### Using Docker Compose (Recommended)

```bash
# Build and start the service
docker-compose up -d

# Check logs
docker-compose logs -f

# Stop the service
docker-compose down
```

#### Using Docker directly

```bash
# Build the image
docker build -t rice-disease-api .

# Run the container
docker run -p 5000:5000 -v $(pwd)/best_model.pth:/app/best_model.pth:ro rice-disease-api
```

### 3. Cloud Deployment

#### Render (Recommended for this project)

1. Push your code to GitHub
2. Connect your GitHub repository to Render
3. Use the following settings:
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn app:app --bind 0.0.0.0:$PORT --workers 1 --timeout 120`
   - **Environment Variables**:
     - `PYTHONUNBUFFERED=1`
     - `FLASK_ENV=production`

Or use the included `render.yaml` file for automatic configuration.

**Note**: The model file (94MB) may cause longer deployment times. Consider using model compression or external storage for production.

#### Heroku

1. Create a `Procfile`:
```
web: gunicorn app:app --bind 0.0.0.0:$PORT --workers 1 --timeout 120
```

2. Deploy:
```bash
heroku create your-app-name
git add .
git commit -m "Initial deployment"
git push heroku main
```

#### Google Cloud Run

1. Build and deploy:
```bash
gcloud builds submit --tag gcr.io/YOUR_PROJECT_ID/rice-disease-api
gcloud run deploy --image gcr.io/YOUR_PROJECT_ID/rice-disease-api --platform managed
```

#### AWS Lambda

1. Create a `lambda_function.py`:
```python
import json
from app import app

def lambda_handler(event, context):
    with app.test_client() as client:
        # Handle the event and return response
        pass
```

2. Package and deploy using AWS CLI or SAM.

## API Endpoints

### Health Check
```bash
curl http://localhost:5000/health
```

### Get Available Classes
```bash
curl http://localhost:5000/classes
```

### Predict from File Upload
```bash
curl -X POST -F "image=@your_image.jpg" http://localhost:5000/predict
```

### Predict from Base64
```bash
curl -X POST -H "Content-Type: application/json" \
  -d '{"image":"base64_encoded_image_data"}' \
  http://localhost:5000/predict_base64
```

## Testing

Run the test suite:
```bash
python api_test.py
```

## Environment Variables

- `FLASK_ENV`: Set to `production` for production deployment
- `PYTHONUNBUFFERED`: Set to `1` for better logging in containers

## Production Considerations

### 1. Security
- Use HTTPS in production
- Implement rate limiting
- Add authentication if needed
- Validate file uploads

### 2. Performance
- Use a production WSGI server (Gunicorn, uWSGI)
- Enable caching
- Use a CDN for static files
- Monitor resource usage

### 3. Monitoring
- Add logging to a file or external service
- Set up health checks
- Monitor API response times
- Track error rates

### 4. Scaling
- Use load balancers
- Deploy multiple instances
- Use container orchestration (Kubernetes, Docker Swarm)

## Example Production Setup with Gunicorn

1. Install Gunicorn:
```bash
pip install gunicorn
```

2. Create `wsgi.py`:
```python
from app import app

if __name__ == "__main__":
    app.run()
```

3. Run with Gunicorn:
```bash
gunicorn -w 4 -b 0.0.0.0:5000 wsgi:app
```

## Troubleshooting

### Common Issues

1. **Model not found error**
   - Ensure `best_model.pth` exists in the project directory
   - Check file permissions

2. **Port already in use**
   - Change the port in `app.py`
   - Kill existing processes using the port

3. **Memory issues**
   - Reduce batch size in training
   - Use smaller model architecture
   - Increase system memory

4. **CUDA out of memory**
   - Use CPU instead of GPU
   - Reduce model size
   - Use gradient checkpointing

### Logs

Check logs for debugging:
```bash
# Docker logs
docker-compose logs

# Application logs
tail -f app.log
```

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review the API documentation at `http://localhost:5000`
3. Run the test suite to verify functionality 