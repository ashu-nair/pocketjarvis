# Keeping bot.py alive in the background

Android aggressively kills background processes to save battery. Two things
fix this:

## 1. Termux wake lock

Before starting the bot, acquire a wake lock so Termux itself doesn't get
frozen:

```bash
termux-wake-lock
python bot.py
```

## 2. Disable battery optimization for Termux

Android Settings → Apps → Termux → Battery → set to **Unrestricted**
(exact wording varies by Android version/manufacturer - look for
"battery optimization" or "background activity").

## 3. Auto-start on boot with Termux:Boot

1. Install **Termux:Boot** from F-Droid (same source as Termux)
2. Open it once so Android registers it
3. Create `~/.termux/boot/start-pocket-jarvis.sh`:

```bash
#!/data/data/com.termux/files/usr/bin/bash
termux-wake-lock
cd ~/pocket-jarvis
python bot.py >> ~/pocket-jarvis/bot.log 2>&1
```

4. Make it executable:
```bash
chmod +x ~/.termux/boot/start-pocket-jarvis.sh
```

Now the bot starts automatically whenever the phone reboots.

## Running it persistently without a reboot

If you just want it running continuously right now without waiting for a
reboot, run it inside a Termux session and detach cleanly using `tmux`:

```bash
pkg install tmux -y
tmux new -s jarvis
python bot.py
# Press Ctrl+B then D to detach - it keeps running
# Reattach later with: tmux attach -t jarvis
```
