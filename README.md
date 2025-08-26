# Toxic Comment Moderation

This project is a production-grade MLOps system. This end-to-end machine learning application includes
model experimentation, versioning, automated testing, deployment on AWS, and live monitoring for
toxic comment moderation. This mult-component application includes:
    * Experiment Tracking & Model Registry: A system to log experiment parameters/metrics and manage model versions
    * ML Model Backend: A `FastAPI` application to serve your registered model
    * Persistent Data Store: A cloud-native base SQL for storing logs and feedback
    * Frontend Interface: A `Streamlit` application for interacting wth the model
    * Model monitoring Dashboard: to detect latency, target drift, user feedback on predictions to calculate accuracy
    * CI/CD Pipeline: An automated workflow to test and validate code changes

## Project Structure

```
├── api/                  # FastAPI backend
│   ├── main.py
│   ├── __init__.py
│   ├── Dockerfile
│   ├── requirements.txt
│   └── toxicity_model.pkl
│
├── monitoring/           # Streamlit dashboard
│   ├── app.py
│   ├── Dockerfile
│   ├── requirements.txt
|   |__ test_dashboard.py # test for Streamlit dashboard
│
├── frontend
│   ├── app.py
│    
├── data/
│   ├── train.csv
│ 
├── .github/workflows/ci.yml
│ 
├── mlruns/
│   ├── models
│      ├── ToxicCommentModel
│      ├── ToxicCommentModel_v1
│      ├── ToxicCommentModel_v2
│    
├── tests/
│   ├── test_api.py
│   ├── test_database.py
│   ├── test_model.py
│      
└── README.md
```
Setup Instructions
GIT

Deployment Steps
Create EC2 instances
Create AWS RDS database and connect to EC2 instance

Build and Run Docker
build -t toxicapi ./api
build -t monitordash ./monitoring

Example Requests by User
FastAPI get & predict

## API Endpoints (FastAPI)

| Endpoint    | Method | Description |
|-------------|--------|-------------|
| `/health`   | GET    | Health check – returns `{"status": "ok"}` |
| `/predict`  | POST   | Accepts JSON input `{"text": "...", "true_status": "..."}` and returns predicted status |

Example:
```json
{
  "text": "It is a lovely day!",
  "true_sentiment": 0          # non-toxic comment 
}
```

Returns:
```json
{
  "predicted_sentiment": 0    # predicted as non-toxic comment
}
```

### Running the System with Docker

1. Build the Containers

   ```bash
   docker-compose build --no-cache
   docker-compose up -d
   ```

Once running, access the UI at: http://<your_public_ip>:8000 and http://<your_public_ip>:8501

### Hosting on AWS EC2 

This project has also been connected to an **AWS EC2 instance**.

#### Steps to Deploy:

1. **Launch an EC2 instance** (e.g., Ubuntu).
2. **Add Security Groups** for ports `8000` and `8501` in your EC2 Security Group.
3. **SSH into your instance**:
   ```bash
   ssh -i /path/to/your-key.pem ubuntu@<your-ec2-public-ip>
   ```

4. **Clone the repo and navigate into the directory**:
   ```bash
   git clone https://github.com/daggidags/Final-Project-mlops-toxic-comment-moderation.git
   cd Final-Project-mlops-toxic-comment-moderation
   ```
5. Build the Containers

   ```bash
   docker-compose build --no-cache
   docker-compose up -d
   ```
