# Toxic Comment Moderation

This project is a production-grade MLOps system. This end-to-end machine learning application includes
model experimentation, versioning, automated testing, deployment on AWS, and live monitoring for
toxic comment moderation. This mult-component application includes:  
Unordered sub-list.
- Experiment Tracking & Model Registry: A system to log experiment parameters/metrics and manage model versions using MLFlow. Accuracy and F1 scores are used to evaluate the three models (ToxicCommentModel, ToxicCommentModel_v1, and ToxicCommentModel_v2)
- ML Model Backend: A `FastAPI` application to serve your registered model
- Persistent Data Store: A cloud-native base SQL for storing logs and feedback
- Frontend Interface: A `Streamlit` application for interacting wth the model
- Model monitoring Dashboard: to detect latency, target drift, user feedback on predictions to calculate accuracy
- CI/CD Pipeline: An automated workflow to test and validate code changes
- Unit Tests: for both API and monitoring dashboard

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

## API Endpoints (FastAPI)

| Endpoint    | Method | Description |
|-------------|--------|-------------|
| `/health`   | GET    | Health check – returns `{"status": "ok"}` |
| `/predict`  | POST   | Accepts JSON input `{"text": "...", "true_status": "..."}` and returns predicted status |

Example:
```json
{
  "text": "It is a lovely day!",
  "true_status": 0          # non-toxic comment 
}
```

Returns:
```json
{
  "predicted_status": 0    # predicted as non-toxic comment
}
```

### Hosting on AWS EC2 

This project has been connected to an **AWS EC2 instance** and **AWS RDS Database**.

#### Steps to Deploy:

1. **Runs Models and the MLFlow UI**
   
   ```bash
   python src/train.py 
   python -m mlflow ui
   ```
   Open MLFlow → http://127.0.0.1:5000/#/experiments

   <img width="1190" height="606" alt="image" src="https://github.com/user-attachments/assets/5e14bb85-ba21-4193-a326-37ed5544bd67" />


3. **Launch an EC2 instance** (e.g., Ubuntu).
4. **Add Security Groups** for ports `8000` and `8501` in your EC2 Security Group.
5. **SSH into your instance**:
   ```bash
   ssh -i /path/to/your-key.pem ubuntu@<your-ec2-public-ip>
   ```

6. **Clone the repo and navigate into the directory**:
   ```bash
   git clone https://github.com/<your-username>/<your-repo>.git
   cd Final-Project-mlops-toxic-comment-moderation
   ```
7. **Build and Start the Containers**

   ```bash
   docker-compose build --no-cache
   docker-compose up -d
   ```
8. **Access the Applications:**

*FastAPI* → http://<ec2-public-ip>:8000/docs

*Streamlit Dashboard* → http://<ec2-public-ip>:8501

7. **Access the AWS RDS Database**
   
   ```bash
   psql -h toxicity-db.cdowqssegxo6.us-east-1.rds.amazonaws.com -U postgres -p 5432
   ```

9. **Access the EC2 instance in Bash**

   ```bash
   ssh-keygen -t rsa -b 4096 -C "github-actions"
   scp -i toxicity-key.pem -r ~/ ec2-user@44.210.126.9:/home/ec2-user/
   ```

