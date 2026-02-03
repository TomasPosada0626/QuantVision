# FAQ - Stock Anomaly Detector

## General Usage

**What does this app do?**
Detects anomalies in historical stock prices using statistical and machine learning methods.

**Which tickers can I analyze?**
All tickers supported by Yahoo Finance. Example: AAPL, MSFT, GOOGL, AMZN, TSLA, etc.

**Can I use my own data?**
Yes, you can upload your CSV file from the sidebar.

**How do I export results?**
You can download processed data as CSV and anomaly plots as PNG images.

**Which anomaly detection methods are available?**
Z-Score, Isolation Forest, DBSCAN, Prophet, Rolling Quantile.

**How do I adjust method parameters?**
Use the sidebar controls to modify thresholds, window size, quantiles, etc.

## Installation & Deployment

**How do I install the project?**
1. Clone the repository.
2. Install dependencies with `pip install -r requirements.txt`.
3. Run the app with `streamlit run src/app.py`.

**How do I use Docker?**
1. Build the image with `docker build -t stock-anomaly-detector .`
2. Run the container with `docker run -p 8501:8501 stock-anomaly-detector`

## Security & Sessions

**How does login and registration work?**
You must create an account with name, username, email, and a strong password. Credentials are stored in SQLite.

**How do I log out?**
Use the "Logout" button in the app.

## Common Issues

**Data for a ticker is not downloading.**
Check the ticker name, your internet connection, and the date range.

**Error uploading CSV.**
Make sure your file has date and price columns (Date, Close).

**Image export not working.**
Ensure your environment supports Plotly >=5.0 and there are no errors in the plot.

## Contact & Collaboration

For suggestions, issues, or collaboration, open an issue on GitHub or contact the author.
