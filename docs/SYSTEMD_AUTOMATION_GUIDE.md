# Systemd Automation Guide

## Overview

The Aurora trading system uses systemd user services for automated daily operations. This guide covers the complete automation setup, monitoring, and troubleshooting.

## Service Architecture

### Core Services

#### 1. `paper-preflight.service`
- **Purpose**: Pre-market validation and system health checks
- **Schedule**: 07:30 CT daily (weekdays only)
- **Duration**: ~2-3 minutes
- **Dependencies**: None
- **Critical**: Yes (blocks trading if fails)

#### 2. `paper-trading.service`
- **Purpose**: Main trading execution and position management
- **Schedule**: 08:00 CT daily (weekdays only)
- **Duration**: ~5-10 minutes
- **Dependencies**: `paper-preflight.service`
- **Critical**: Yes (core trading function)

#### 3. `paper-status.service`
- **Purpose**: Intraday monitoring and health checks
- **Schedule**: Every hour during market hours (09:00-15:00 CT)
- **Duration**: ~30 seconds
- **Dependencies**: None
- **Critical**: No (informational only)

#### 4. `paper-eod.service`
- **Purpose**: End-of-day reporting and position reconciliation
- **Schedule**: 15:15 CT daily (weekdays only)
- **Duration**: ~2-3 minutes
- **Dependencies**: `paper-trading.service`
- **Critical**: Yes (daily reporting)

#### 5. `paper-data-fetch.service`
- **Purpose**: Fetch next-day market data
- **Schedule**: 16:00 CT daily (weekdays only)
- **Duration**: ~3-5 minutes
- **Dependencies**: None
- **Critical**: Yes (data pipeline)

### Timer Configuration

All services use systemd timers with the following schedule:

```ini
[Timer]
OnCalendar=Mon..Fri 07:30:00 America/Chicago  # Preflight
OnCalendar=Mon..Fri 08:00:00 America/Chicago  # Trading
OnCalendar=Mon..Fri 09:00:00 America/Chicago  # Status (hourly)
OnCalendar=Mon..Fri 15:15:00 America/Chicago  # EOD
OnCalendar=Mon..Fri 16:00:00 America/Chicago  # Data fetch
```

## Service Files

### Location
All service files are located in `~/.config/systemd/user/`:

```
~/.config/systemd/user/
├── paper-preflight.service
├── paper-preflight.timer
├── paper-trading.service
├── paper-trading.timer
├── paper-status.service
├── paper-status.timer
├── paper-eod.service
├── paper-eod.timer
├── paper-data-fetch.service
└── paper-data-fetch.timer
```

### Service Template

Each service follows this template:

```ini
[Unit]
Description=Paper Trading [Service Name]
After=network.target

[Service]
Type=oneshot
WorkingDirectory=/home/Jennifer/secure/trader
Environment=PATH=/usr/local/bin:/usr/bin:/bin
Environment=PYTHONUNBUFFERED=1
Environment=TZ=America/Chicago
ExecStart=/bin/bash -lc 'source ~/.config/paper-trading.env && ./daily_paper_trading.sh [mode]'
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=default.target
```

### Timer Template

Each timer follows this template:

```ini
[Unit]
Description=Paper Trading [Service Name] Timer
Requires=paper-[service].service

[Timer]
OnCalendar=[schedule]
Persistent=true

[Install]
WantedBy=timers.target
```

## Environment Configuration

### Persistent Environment
Environment variables are stored in `~/.config/paper-trading.env`:

```bash
# Alpaca API Configuration
APCA_API_KEY_ID="your_api_key"
APCA_API_SECRET_KEY="your_secret_key"
APCA_API_BASE_URL="https://paper-api.alpaca.markets"

# Trading Configuration
IS_PAPER_TRADING="true"
BROKER_ENDPOINT="https://paper-api.alpaca.markets"

# System Configuration
PYTHONUNBUFFERED=1
TZ=America/Chicago
```

### Security
- File permissions: `600` (owner read/write only)
- Auto-loaded by all services
- Never committed to git

## Monitoring and Management

### Service Status
```bash
# Check all service status
systemctl --user status paper-*

# Check specific service
systemctl --user status paper-preflight.service

# Check timer status
systemctl --user list-timers paper-*
```

### Service Control
```bash
# Start/stop services
systemctl --user start paper-preflight.service
systemctl --user stop paper-preflight.service

# Restart services
systemctl --user restart paper-preflight.service

# Enable/disable services
systemctl --user enable paper-preflight.service
systemctl --user disable paper-preflight.service
```

### Log Monitoring
```bash
# View recent logs
journalctl --user -u paper-* --since "1 hour ago"

# Follow live logs
journalctl --user -u paper-* -f

# View specific service logs
journalctl --user -u paper-preflight.service -f
```

### Monitoring Script
Use the provided monitoring script:

```bash
# Check system status
./monitor_paper_trading.sh

# Output example:
# ✅ paper-preflight.timer: active (next: Mon 2025-09-09 07:30:00 CDT)
# ✅ paper-trading.timer: active (next: Mon 2025-09-09 08:00:00 CDT)
# ✅ paper-status.timer: active (next: Mon 2025-09-09 09:00:00 CDT)
# ✅ paper-eod.timer: active (next: Mon 2025-09-09 15:15:00 CDT)
# ✅ paper-data-fetch.timer: active (next: Mon 2025-09-09 16:00:00 CDT)
```

## Troubleshooting

### Common Issues

#### 1. Service Fails to Start
**Symptoms**: Service shows "failed" status
**Causes**: 
- Missing environment variables
- Incorrect file paths
- Permission issues

**Solutions**:
```bash
# Check service logs
journalctl --user -u paper-preflight.service -n 50

# Verify environment
source ~/.config/paper-trading.env
echo $APCA_API_KEY_ID

# Check file permissions
ls -la ~/.config/paper-trading.env
ls -la daily_paper_trading.sh
```

#### 2. Timer Not Scheduled
**Symptoms**: Timer shows "inactive" status
**Causes**:
- Timer not enabled
- Incorrect schedule format
- System timezone issues

**Solutions**:
```bash
# Enable timer
systemctl --user enable paper-preflight.timer

# Check timer schedule
systemctl --user list-timers paper-*

# Verify timezone
timedatectl
```

#### 3. Environment Not Loaded
**Symptoms**: API calls fail with authentication errors
**Causes**:
- Environment file not sourced
- Incorrect file permissions
- Missing variables

**Solutions**:
```bash
# Check environment file
cat ~/.config/paper-trading.env

# Test environment loading
source ~/.config/paper-trading.env
python -c "import os; print(os.getenv('APCA_API_KEY_ID'))"

# Fix permissions if needed
chmod 600 ~/.config/paper-trading.env
```

#### 4. User Lingering Issues
**Symptoms**: Services don't run when user is logged out
**Causes**:
- User lingering not enabled
- Systemd user session not persistent

**Solutions**:
```bash
# Enable user lingering
sudo loginctl enable-linger $USER

# Verify lingering status
loginctl show-user $USER | grep Linger

# Restart user services
systemctl --user daemon-reload
```

### Advanced Troubleshooting

#### Service Dependencies
```bash
# Check service dependencies
systemctl --user list-dependencies paper-trading.service

# Check service conflicts
systemctl --user list-conflicts paper-trading.service
```

#### Resource Monitoring
```bash
# Check system resources
htop

# Check disk space
df -h

# Check memory usage
free -h
```

#### Network Issues
```bash
# Test API connectivity
curl -H "APCA-API-KEY-ID: $APCA_API_KEY_ID" \
     -H "APCA-API-SECRET-KEY: $APCA_API_SECRET_KEY" \
     https://paper-api.alpaca.markets/v2/account

# Check DNS resolution
nslookup paper-api.alpaca.markets
```

## Maintenance

### Regular Maintenance

#### Weekly
- [ ] Review service logs for errors
- [ ] Check timer schedules are correct
- [ ] Verify environment variables are current
- [ ] Test manual service execution

#### Monthly
- [ ] Rotate API keys if needed
- [ ] Update service configurations
- [ ] Review and clean old logs
- [ ] Test emergency procedures

#### Quarterly
- [ ] Review and update service schedules
- [ ] Test complete system restart
- [ ] Verify backup and recovery procedures
- [ ] Update documentation

### Log Management

#### Log Rotation
```bash
# Clean old journal logs
journalctl --user --vacuum-time=30d

# Archive important logs
journalctl --user -u paper-* --since "1 month ago" > logs/systemd_archive_$(date +%Y%m).log
```

#### Log Analysis
```bash
# Find errors in logs
journalctl --user -u paper-* | grep -i error

# Count service executions
journalctl --user -u paper-* | grep -c "Started Paper Trading"

# Check service performance
journalctl --user -u paper-* | grep -E "(Started|Finished)"
```

## Security Considerations

### Service Security
- Services run as user (not root)
- Environment variables stored securely
- API keys never logged
- File permissions properly set

### Network Security
- All API calls use HTTPS
- API keys rotated regularly
- No sensitive data in logs
- Firewall rules as needed

### Access Control
- Services only accessible to owner
- Environment file permissions 600
- No shared access to trading system
- Regular security audits

## Performance Optimization

### Service Performance
- Services use minimal resources
- Proper timeout settings
- Efficient logging
- Resource monitoring

### System Performance
- Regular system updates
- Disk space monitoring
- Memory usage tracking
- Network latency monitoring

### Scaling Considerations
- Services designed for single user
- Easy to extend for multiple users
- Resource limits configurable
- Monitoring scales with system

## Backup and Recovery

### Service Configuration Backup
```bash
# Backup service files
tar -czf systemd_backup_$(date +%Y%m%d).tar.gz ~/.config/systemd/user/

# Backup environment
cp ~/.config/paper-trading.env ~/.config/paper-trading.env.backup
```

### Service Recovery
```bash
# Restore service files
tar -xzf systemd_backup_20250908.tar.gz -C ~/

# Reload systemd
systemctl --user daemon-reload

# Restart services
systemctl --user restart paper-*
```

### Complete System Recovery
```bash
# Restore from git
git checkout paper-launch-d0

# Restore environment
cp ~/.config/paper-trading.env.backup ~/.config/paper-trading.env

# Restart all services
systemctl --user restart paper-*
```
