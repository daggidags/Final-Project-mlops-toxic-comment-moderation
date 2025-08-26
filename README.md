Toxic Comment Moderation

This project is a production-grade MLOps system. This end-to-end machine learning application includes
model experimentation, versioning, automated testing, deployment on AWS, and live monitoring for
toxic comment moderation. This mult-component application includes:
    * Experiment Tracking & Model Registry: A system to log experiment parameters/metrics and manage model versions
    * ML Model Backend: A FastAPI application to serve your registered model
    * Persistent Data Store: A cloud-native base SQL for storing logs and feedback
    * Frontend Interface: A streamlit application for interacting wth the model
    * Model monitoring Dashboard: to detect latency, target drift, user feedback on predictions to calculate accuracy
    * CI/CD Pipeline: An automated workflow to test and validate code changes

Project Structure
```bash 
tree
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