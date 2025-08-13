# 🔔 Discord Notification Setup Guide

This guide will help you set up Discord notifications for your enhanced trading system.

## 📋 Prerequisites

- A Discord account
- A Discord server where you want to receive notifications
- Admin permissions on the server (to create webhooks)

## 🚀 Step-by-Step Setup

### 1. Create a Discord Webhook

1. **Open Discord** and navigate to your server
2. **Right-click** on the channel where you want notifications
3. Select **"Edit Channel"**
4. Go to **"Integrations"** tab
5. Click **"Create Webhook"**
6. Give it a name like **"Trading Bot"**
7. Click **"Copy Webhook URL"** (save this URL!)

### 2. Configure the Trading System

1. **Edit the Discord config file:**
   ```bash
   nano config/notifications/discord_config.json
   ```

2. **Replace the placeholder with your webhook URL:**
   ```json
   {
     "webhook_url": "https://discord.com/api/webhooks/YOUR_WEBHOOK_ID/YOUR_WEBHOOK_TOKEN",
     "bot_name": "Trading Bot",
     "bot_avatar": "https://cdn.discordapp.com/emojis/📈.png",
     "enabled": true,
     "notifications": {
       "startup": true,
       "trades": true,
       "daily_summary": true,
       "errors": true,
       "cron_execution": true
     }
   }
   ```

### 3. Test the Configuration

Run the trading system to test Discord notifications:

```bash
python enhanced_paper_trading.py --daily
```

## 📊 Notification Types

### 🚀 Startup Notification
- **When**: System starts up
- **Content**: Initial capital, active strategies, target returns
- **Color**: Green

### 💰 Trade Notifications
- **When**: Trade is executed
- **Content**: Symbol, action, size, price, value, regime
- **Color**: Green (BUY) / Red (SELL)

### 📈 Daily Summary
- **When**: End of daily trading
- **Content**: Total return, capital, Sharpe ratio, max drawdown
- **Color**: Green (profit) / Red (loss)

### ✅ Cron Execution
- **When**: Automated trading completes
- **Content**: Execution status and details
- **Color**: Green (success) / Red (failure)

### ❌ Error Notifications
- **When**: System encounters errors
- **Content**: Error message and context
- **Color**: Red

## 🔧 Configuration Options

### Basic Settings
```json
{
  "webhook_url": "YOUR_WEBHOOK_URL",
  "bot_name": "Trading Bot",
  "bot_avatar": "https://cdn.discordapp.com/emojis/📈.png",
  "enabled": true
}
```

### Notification Preferences
```json
{
  "notifications": {
    "startup": true,        // System startup
    "trades": true,         // Individual trades
    "daily_summary": true,  // Daily performance
    "errors": true,         // Error alerts
    "cron_execution": true  // Cron job status
  }
}
```

## 🎨 Customization

### Change Bot Name
```json
{
  "bot_name": "My Trading Bot"
}
```

### Change Bot Avatar
```json
{
  "bot_avatar": "https://your-custom-avatar-url.png"
}
```

### Disable Specific Notifications
```json
{
  "notifications": {
    "startup": true,
    "trades": false,        // Disable trade notifications
    "daily_summary": true,
    "errors": true,
    "cron_execution": true
  }
}
```

## 🔒 Security Best Practices

1. **Keep your webhook URL private** - don't share it publicly
2. **Use a dedicated channel** for trading notifications
3. **Regularly rotate webhook URLs** if needed
4. **Monitor webhook usage** in Discord server settings

## 🐛 Troubleshooting

### Notifications Not Working?

1. **Check webhook URL:**
   - Ensure it's copied correctly
   - Verify the webhook still exists in Discord

2. **Check configuration:**
   ```bash
   cat config/notifications/discord_config.json
   ```

3. **Check logs:**
   ```bash
   tail -f logs/trading_bot.log
   ```

4. **Test webhook manually:**
   ```bash
   curl -H "Content-Type: application/json" \
        -X POST \
        -d '{"content":"Test message"}' \
        YOUR_WEBHOOK_URL
   ```

### Common Issues

- **"Discord notifications disabled"**: Check `enabled` field in config
- **"Webhook URL not configured"**: Replace placeholder URL
- **"Failed to send notification"**: Check internet connection and webhook URL

## 📱 Mobile Notifications

To receive notifications on your phone:

1. **Enable Discord mobile notifications** in your phone settings
2. **Set up Discord mobile app** notifications for the trading channel
3. **Configure notification preferences** in Discord app settings

## 🔄 Automation with Cron

When using cron jobs, add the `--cron` flag:

```bash
# In your crontab
0 9 * * 1-5 cd /path/to/trader && python enhanced_paper_trading.py --cron
```

This will send cron execution notifications to Discord.

## 📊 Example Notifications

### Startup Notification
```
🚀 Trading System Started
💰 Initial Capital: $100,000
📊 Strategies: 4 active
🎯 Target Return: 65%+ annually
```

### Trade Notification
```
🔴 Trade Executed
Symbol: SPY
Action: SELL
Size: 21.78
Price: $642.69
Value: $14,000
Regime: chop (50.0%)
```

### Daily Summary
```
📉 Daily Trading Summary
📊 Total Return: -89.98%
💰 Current Capital: $-8,897,660
📈 Sharpe Ratio: 0.00
📉 Max Drawdown: 0.00%
🔄 Total Trades: 1
🎯 Regime: chop (91.7%)
```

## 🎯 Next Steps

1. **Configure your webhook URL**
2. **Test with a daily run**
3. **Customize notification preferences**
4. **Set up mobile notifications**
5. **Monitor and adjust as needed**

---

**Your Discord notifications are now ready to keep you informed about your trading system's performance!** 🎉
