!pip install telethon textblob scikit-learn pytz
!pip install python-dotenv
import os
from telethon import TelegramClient
from telethon.sessions import MemorySession  # Use MemorySession to avoid SQLite issues
import pandas as pd
import re
from datetime import datetime, timedelta
import asyncio

# Set your Telegram API credentials
api_id = 'Your API ID'  # Replace with your own API ID
api_hash = 'Your Hash Code'  # Replace with your own API hash
phone_number = 'Your Phone Number'  # Replace with your phone number

channels = ['@StockMarketChat', '@WallStreetBets']  # Add stock-related channels

# Clean the text function
def clean_text(text):
    if not text:  # Check if the text is None or empty
        return ""
    text = str(text)  # Ensure text is a string
    text = re.sub(r"http\S+", "", text)  # Remove URLs
    text = re.sub(r"@\S+|#\S+", "", text)  # Remove mentions and hashtags
    text = re.sub(r"[^A-Za-z0-9\s]", "", text)  # Remove special characters
    return text.lower()

# Scrape messages function
async def scrape_channel(client, channel, days_to_scrape=30):
    messages = []
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_to_scrape)

    async for message in client.iter_messages(channel, limit=400, offset_date=start_date):
        messages.append({
            'channel': channel,
            'date': message.date.strftime('%Y-%m-%d %H:%M:%S'),
            'text': message.text
        })
    return messages

# Main function to scrape data
async def main():
    # Initialize the Telegram client with MemorySession to avoid SQLite issues
    client = TelegramClient(MemorySession(), api_id, api_hash)

    # Start the client and ensure it's properly connected
    await client.start(phone=phone_number)

    all_messages = []
    for channel in channels:
        channel_messages = await scrape_channel(client, channel)
        all_messages.extend(channel_messages)

    # Clean the text
    for message in all_messages:
        message['clean_text'] = clean_text(message['text'])

    # Convert to DataFrame and save to CSV
    df = pd.DataFrame(all_messages)
    df.to_csv('telegram_stock_data.csv', index=False)

    await client.disconnect()  # Disconnect the client after finishing

# Run the main function directly in the environment
await main()  # Directly use 'await' instead of asyncio.run()
