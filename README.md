# Telegram-Stock-Sentiment-Scraper-Analyzer

This project scrapes messages from specified Telegram channels related to the stock market, cleans and processes the messages, performs sentiment analysis, and uses machine learning to classify the messages based on their relevance to trading.

---
## Features

- Scrape messages from Telegram channels.
- Clean the extracted text by removing URLs, mentions, hashtags, and special characters.
- Perform sentiment analysis using the `TextBlob` library.
- Classify messages using a machine learning model (Random Forest).
- Save processed data in CSV format for further analysis.

---

## Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/telegram-stock-analyzer.git
   cd telegram-stock-analyzer
2.**Install Required Libraries**: Ensure Python 3.8+ is installed. Install the necessary libraries:  

                  pip install telethon textblob scikit-learn pandas numpy pytz python-dotenv
                   
3.**Set Up Telegram API Credentials:**

Obtain your api_id and api_hash from the Telegram API.
Replace the placeholders in the script (api_id, api_hash, and phone_number) with your credentials.

##**Usage**

**1.Scraping Messages:**

Run the main function to scrape messages from the specified Telegram channels.
Messages will be saved to a CSV file named telegram_stock_data.csv.

                              python telegram_scraper.py

**2.Sentiment Analysis:**

The script uses the TextBlob library to calculate sentiment polarity for each message. Messages are assigned a sentiment score from -1 (negative) to 1 (positive).

**3.Message Classification:**

The RandomForestClassifier is used to classify messages based on whether they are related to trading or stocks.
Evaluation Metrics:

4.**After training the machine learning model, the following metrics are displayed:**

Accuracy

Precision

Recall

F1 Score

**5.Download the Processed CSV File:**


The processed CSV files can be downloaded directly:
telegram_stock_data.csv
cleaned_telegram_stock_data.csv



