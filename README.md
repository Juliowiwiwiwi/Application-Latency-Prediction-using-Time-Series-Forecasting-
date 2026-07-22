# Application Latency Prediction in Time-Series [WORK IN PROGRESS]

An AIOps project that predicts future application latency using time-series forecasting and alerts engineers before performance issues impact users.

## Overview

The system analyzes historical application telemetry such as:

- CPU Usage
- Memory Usage
- Request Traffic
- Database Response Time
- Error Rate
- Application Latency

Using this data, a machine learning model forecasts application latency **30 minutes ahead**.

If the predicted latency exceeds a predefined SLA threshold, the system generates an alert, allowing engineers to take preventive action before users experience slowdowns.

---

## 🔄 Workflow

```text
Telemetry Data
      │
      ▼
 ML Forecasting Model
      │
      ▼
Predict Latency (30 mins ahead)
      │
      ▼
Above SLA Threshold?
   │           │
  Yes          No
   │           │
   ▼           ▼
Send Alert   Continue Monitoring
```

---

## 🎯 Features

- Time-series latency prediction
- Predicts latency 30 minutes in advance
- Threshold-based alerting
- Performance visualization
- Extensible architecture for future AIOps capabilities

---

## 🔮 Future Enhancements

- Root cause analysis for latency spikes
- AI-generated incident reports
- Recommended remediation actions
- Interactive monitoring dashboard

---

## 📈 Project Status

🚧 Work in Progress