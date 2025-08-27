# 📦 Crypto Watchlist Telegram Bot

A Telegram bot for tracking crypto coins, marking them as haram/halal, and managing a Supabase-backed watchlist.

## 🚀 Features
- ✅ Add coins to watchlist
- 🗑️ Remove coins and log removal
- ⚠️ Mark coins as haram or halal
- 🔐 Admin-only access
- 🔄 Persistent memory via Supabase
- 🌐 Flask keep-alive for Render deployment

## ⚙️ Environment Variables
```env
BOT_TOKEN=your_telegram_bot_token
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your_supabase_service_role_key
