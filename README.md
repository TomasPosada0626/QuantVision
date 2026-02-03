# Stock Anomaly Detector

<p align="center">
  <b>Professional, advanced anomaly detection in historical stock prices</b><br>
  <i>Python, Pandas, Numpy, Matplotlib, Seaborn, Plotly, Scikit-learn, Prophet, Streamlit, Pytest, Data Science, Finance, Machine Learning</i>
</p>
<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8%2B-blue" />
  <img src="https://img.shields.io/badge/Pandas-2.x-blue" />
  <img src="https://img.shields.io/badge/Numpy-1.x-blue" />
  <img src="https://img.shields.io/badge/Matplotlib-3.x-blue" />
  <img src="https://img.shields.io/badge/Seaborn-0.x-blue" />
  <img src="https://img.shields.io/badge/Plotly-5.x-blue" />
  <img src="https://img.shields.io/badge/scikit--learn-1.x-blue" />
  <img src="https://img.shields.io/badge/Prophet-1.x-blue" />
  <img src="https://img.shields.io/badge/Streamlit-1.x-red" />
  <img src="https://img.shields.io/badge/Pytest-7.x-green" />
  <img src="https://img.shields.io/badge/Data-Science-grey" />
  <img src="https://img.shields.io/badge/Finance-grey" />
  <img src="https://img.shields.io/badge/Machine-Learning-purple" />
  <img src="https://img.shields.io/badge/License-MIT-brightgreen" />
  <img src="https://img.shields.io/badge/build-passing-brightgreen" />
  <img src="https://img.shields.io/badge/License-MIT-yellow.svg" />
  <img src="https://img.shields.io/badge/python-3.9%2B-blue" />
  <img src="https://img.shields.io/badge/docker-ready-blue" />
</p>

---

A professional Python project for detecting anomalies in historical stock prices using advanced statistical and machine learning methods. Designed for recruiters, data scientists, and finance professionals.

## Table of Contents
- [Project Overview](#project-overview)
- [Advanced Notebooks & Case Studies](#advanced-notebooks--case-studies)
- [Streamlit App Features](#streamlit-app-features-login--session-management)
- [Features](#features)
- [Demo](#demo)
- [Anomaly Detection Methods Explained](#anomaly-detection-methods-explained)
- [Method Comparison Table](#method-comparison-table)
- [Case Study: Real Event Detection](#case-study-real-event-detection)
- [Known Limitations](#known-limitations)
- [FAQ](#frequently-asked-questions-faq)
- [Automated Testing](#automated-testing)
- [Deployment Instructions](#deployment-instructions)
- [Docker Instructions](#docker-instructions)
- [Hardware/Software Requirements](#hardwaresoftware-requirements)
- [Contact & Support](#contact--support)
- [Future Work](#future-work)
- [Advanced Usage Examples](#advanced-usage-examples)
- [Visual Polish: Streamlit App UI Tips](#visual-polish-streamlit-app-ui-tips)
- [Professional Touches](#professional-touches)
- [Screenshots](#screenshots)
- [Contributors](CONTRIBUTORS.md)
- [Changelog](CHANGELOG.md)
- [Security & User Experience](#security--user-experience)

---

## Project Overview
This project provides a professional, advanced anomaly detection system for historical stock prices using Python and machine learning methods. It is designed for recruiters, data scientists, and finance professionals to analyze and detect anomalies in stock data.

## Advanced Notebooks & Case Studies

Explore advanced workflows and real-world case studies in the following Jupyter notebooks:

- [Stock Anomaly Analysis Notebook](notebooks/stock_anomaly_analysis.ipynb):
  - Professional, modular analysis of historical stock prices using multiple anomaly detection methods (Z-Score, Isolation Forest, DBSCAN, Prophet, Rolling Quantile).
  - Includes multi-ticker comparison, benchmarking, and extensibility for new assets or methods.
  - Ready for portfolio and interview use, with clear code and best practices.

- [Deep Learning Anomaly Case Studies Notebook](notebooks/deep_learning_anomaly_case_studies.ipynb):
  - Advanced deep learning techniques (LSTM, autoencoders) for anomaly detection in financial time series.
  - Real-world case studies: COVID-19 crisis, earnings events, and comparison with classical methods.
  - Demonstrates how to extend the project with neural networks and interpret results for professional applications.

See each notebook for step-by-step code, visualizations, and recommendations for portfolio and interview use.

---

## Streamlit App Features (Login & Session Management)

The web app now includes a professional, secure login and registration system:

- **Login/Registration Panel:** Users can log in or register with first name, last name, username, email, and password.
- **Username Availability Check:** Real-time feedback on username availability during registration.
- **Password Validation:** Passwords must be strong (minimum 8 characters, uppercase, lowercase, number, special character) and match confirmation.
- **Session Management:** After successful login, users are automatically logged in and can access the app immediately.
- **Logout Button:** Users can securely log out, which clears their session and returns to the login screen.
- **Consistent Layout:** The app layout remains stable and professional during login/logout transitions.
- **Autocomplete Support:** Registration fields support browser autocomplete for improved user experience.
- **SQLite Integration:** User credentials and sessions are securely stored in a local SQLite database.
- **All feedback and messages are in English for professional presentation.**

These features ensure a recruiter-ready, user-friendly, and secure experience for all users.

---

## Features
- **Interactive Web App**: Built with Streamlit for easy selection of stocks and date ranges.
- **Automated Data Download**: Fetches data from Yahoo Finance and stores it locally for fast access.
- **Exploratory Data Analysis (EDA)**: Advanced visualizations with matplotlib, seaborn, and plotly.
- **Multiple Anomaly Detection Methods**: Z-Score, Isolation Forest, DBSCAN, Prophet.
- **Downloadable Results**: Export detected anomalies and statistics.
- **Jupyter Notebook**: Step-by-step analysis and modeling for reproducibility.
- **Automated Testing**: Unit tests for core anomaly detection functions.
- **Modular Code**: Clean, well-documented, and extensible.

## Demo

### 1. Main Dashboard
![Main Dashboard](./screenshots/1.png)

### 2. Exploratory Data Analysis (EDA)
![EDA](./screenshots/2.png)

### 3. Anomaly Detection Visualization
![Anomaly Detection](./screenshots/3.png)

## Anomaly Detection Methods Explained
- **Z-Score**: Flags data points that deviate more than a set number of standard deviations from the mean. Best for normally distributed data.
- **Isolation Forest**: Uses an ensemble of trees to isolate anomalies. Effective for high-dimensional and non-Gaussian data.
- **DBSCAN**: Clustering-based method that identifies outliers as points not belonging to any cluster. Good for spatial or density-based anomalies.
- **Prophet**: Time series forecasting model; anomalies are detected as large residuals from the predicted trend.
- **Rolling Quantile**: Detects anomalies when the closing price falls outside rolling quantile thresholds (configurable window and quantiles). Useful for identifying outliers in non-stationary time series.
  - Parameters: window size, lower quantile, upper quantile (all configurable in the sidebar).
  - Results are visualized and included in the method comparison/benchmarking table.

## Method Comparison Table

| Method            | Strengths                                      | Weaknesses                                 | Best Use Case                      |
|-------------------|------------------------------------------------|--------------------------------------------|------------------------------------|
| Z-Score           | Simple, fast, interpretable                    | Assumes normality, sensitive to outliers   | Quick checks, normal data          |
| Isolation Forest  | Robust, works with high-dimensional data       | Needs contamination tuning                 | General anomaly detection          |
| DBSCAN            | Finds clusters, density-based                  | Sensitive to parameters, not for all data  | Spatial/density anomalies          |
| Prophet           | Handles trend/seasonality, interpretable       | May miss short-term anomalies              | Time series with clear seasonality |
| Rolling Quantile  | Non-parametric, robust to outliers             | Window/quantile tuning needed              | Extreme value detection            |

## Case Study: Real Event Detection

*Example: Detecting anomalies in AAPL during the COVID-19 market crash (2020)*
- Select AAPL and date range 2019-01-01 to 2021-01-01 in the app.
- Observe spike in anomalies detected by all methods around March 2020.
- Compare which methods are more/less sensitive to the event.

## Known Limitations

- The project relies on historical price data and does not incorporate alternative data sources (news, sentiment, macroeconomic indicators).
- Anomaly labels are unsupervised; there is no ground truth for "true" anomalies, so evaluation is qualitative.
- Deep learning and real-time integration are provided as examples, not as full production pipelines.
- Model hyperparameters are not exhaustively optimized; results may vary by asset or market regime.
- The project does not address market microstructure noise or high-frequency trading anomalies.
- Performance may be limited for very large datasets on low-resource machines.
- Some advanced features (e.g., image export, PDF export, real-time streaming) are planned but not yet implemented.

## Frequently Asked Questions (FAQ)

**Q: What tickers are available?**
A: By default: AAPL, MSFT, GOOGL, AMZN, TSLA, META, NVDA, JPM, V, DIS. You can add more by downloading new data or adding your own CSV files.

**Q: Can I use my own data?**
A: Yes, place your CSV files in the `data/` folder and select them in the app.

**Q: How do I export results?**
A: Use the download button in the app to export anomalies and statistics as CSV. Excel and PDF export are planned for future releases.

**Q: How do I contribute?**
A: See the 'How to Contribute' section above.

**Q: What if I find a bug?**
A: Please open an issue or contact the author for support.

**Q: Is there support for deep learning or real-time data?**
A: Yes, see the advanced notebooks for deep learning examples. Real-time data integration is planned for future updates.

**Q: Can I deploy this project with Docker or on the cloud?**
A: Yes, see the Docker and cloud deployment instructions in the README.

---

## Automated Testing

This project includes automated tests for the main anomaly detection methods (Z-Score, Isolation Forest, Rolling Quantile).

### How to Run Tests
1. Make sure you have all dependencies installed:
   ```bash
   pip install -r requirements.txt
   ```
2. Run all tests using pytest:
   ```bash
   pytest tests/tests.py
   ```

### What is Tested?
- Z-Score anomaly detection (detects outliers in returns)
- Isolation Forest anomaly detection (detects outliers in returns)
- Rolling Quantile anomaly detection (detects outliers in time series)
- Edge cases: no anomalies, all normal data, strong outliers

### Expected Results
- All tests should pass with no errors.
- If a test fails, check the implementation in `src/anomaly_methods.py` or `src/utils.py`.

### Adding More Tests
- Add new test functions to `tests/tests.py` following the examples.
- Use clear assertions and comments for maintainability.

These tests ensure the reliability and correctness of the main anomaly detection methods in your project.

---

## Deployment Instructions

### Local Deployment
1. Clone the repository and install dependencies:
   ```bash
   pip install -r requirements.txt
   streamlit run app.py
   ```

### Docker Deployment
1. Build the Docker image:
   ```bash
   docker build -t stock-anomaly-detector .
   ```
2. Run the container:
   ```bash
   docker run -p 8501:8501 stock-anomaly-detector
   ```

### Cloud Deployment
- You can deploy this app to [Streamlit Cloud](https://streamlit.io/cloud) or any cloud provider supporting Docker.
- For Heroku, use a `Procfile` with `web: streamlit run app.py` and set the port to `$PORT`.

## Docker Instructions

You can easily deploy this project using Docker for reproducible environments and cloud deployment.

### Build the Docker image
```bash
docker build -t stock-anomaly-detector .
```

### Run the Docker container
```bash
docker run -p 8501:8501 stock-anomaly-detector
```

- The app will be available at `http://localhost:8501`.
- All project files and dependencies are included in the container.
- For custom data, mount your local `data/` folder as a volume:
  ```bash
  docker run -p 8501:8501 -v $(pwd)/data:/app/data stock-anomaly-detector
  ```

### Cloud Deployment
- You can deploy this image to any cloud provider supporting Docker (AWS, Azure, GCP, Heroku, Streamlit Cloud).
- For Heroku, use a `Procfile` with `web: streamlit run app.py` and set the port to `$PORT`.

## Hardware/Software Requirements
- Python 3.8+
- 2+ GB RAM recommended for large datasets
- Works on Windows, macOS, Linux, Docker, and most cloud platforms

---

## Contact & Support
**Author:** Tomás Posada  
**Email:** [tomasposada67@gmail.com](mailto:tomasposada67@gmail.com)  
**LinkedIn:** [Your LinkedIn Profile]

For suggestions, improvements, or support, please open an issue in the repository or contact the author directly.

## Future Work
- Add CI/CD pipeline for automated testing and deployment
- Integrate new anomaly detection methods (e.g., LSTM, Autoencoders)
- Support for real-time data streaming
- Integration with databases (SQL/NoSQL)
- More advanced visualizations and dashboards
- Multi-language support
- Performance optimization for very large datasets

## Advanced Usage Examples

### Compare Multiple Tickers and Methods
You can select multiple tickers and anomaly detection methods in the app sidebar. The app will download, analyze, and visualize anomalies for each ticker and method, allowing you to compare results side by side.

### Adjust Parameters
Each method has adjustable parameters (e.g., Z-Score threshold, IForest contamination, DBSCAN eps/min_samples, Rolling Quantile window/quantile). Use the sidebar sliders to fine-tune detection sensitivity.

### Export Results
After analysis, export anomalies for each ticker in CSV or Excel format using the download buttons below each visualization.

### Use with Custom Datasets
Place your own CSV files in the `data/` folder. You can select them by entering the ticker name in the custom ticker field.

### Extend with New Methods
Add new anomaly detection functions in `anomaly_methods.py` or `utils.py`, add tests, and select them in the app for instant comparison.

# Visual Polish: Streamlit App UI Tips
- Use `st.markdown` for section headers and instructions.
- Add `st.sidebar` for parameter selection to declutter main view.
- Use `st.success`, `st.warning`, `st.error` for user feedback.
- Add tooltips to sliders and dropdowns for parameter explanations.
- Use `plotly` for interactive, zoomable anomaly plots.
- Add download/export buttons for results.

# Professional Touches
- Add a project logo or banner at the top of the README.
- Include a `LICENSE` file (MIT recommended for open source).
- Add badges (build, license, Python version, Docker) to README.
- Link to your LinkedIn, portfolio, or relevant publications.
- Add a `CONTRIBUTING.md` for open source collaboration.
- Include a `CHANGELOG.md` for version tracking.

# Future Work Ideas
- Integrate LSTM/autoencoder deep learning anomaly detection.
- Add alerting/email notification for detected anomalies.
- Support for real-time streaming data (e.g., WebSocket, Kafka).
- Add REST API endpoints for programmatic access.
- More granular anomaly explanations (SHAP, LIME, etc.).

## Screenshots

| Multi-Ticker Selection | Interactive Anomaly Plot | Export Results |
|-----------------------|-------------------------|----------------|
| ![Multi-Ticker](screenshots/1.png) | ![Plot](screenshots/2.png) | ![Export](screenshots/3.png) |

## Security & User Experience
- All passwords are securely hashed before storage (SHA-256).
- User sessions are managed with unique session IDs and SQLite backend.
- Registration requires strong passwords and checks for username/email availability.
- All user feedback is clear, in English, and uses professional UI elements (success, error, warning).
- Autocomplete is enabled for registration fields to improve usability.
- Logout button allows users to securely end their session.
- No sensitive data is exposed in logs or UI.
- The app is designed for recruiter-ready, professional presentation and ease of use.
