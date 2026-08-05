# Pocket Jarvis

Turn an old Android phone into a locally-controlled AI agent you can message
from anywhere — no VLAN, no port forwarding, no always-on laptop required.

## How it works

```
You (any device, any network)
        │
        │  Telegram message
        ▼
Telegram Bot API  ◄────── Phone polls every few seconds
        │
        ▼
  bot.py (running in Termux on the phone)
        │
        ▼
  [Phase 2+] Accessibility Service → controls the phone
  [Phase 3+] PocketClaw (local model) → reasons about what to do
```

The phone never needs an open port. It reaches *out* to Telegram's servers
on a timer and checks "anything new for me?" — this works identically on
home wifi, a hotspot, or any network with internet access.

## Project phases

- **Phase 1 (this commit)** — plumbing only. Send a Telegram message, phone
  echoes it back. Proves the pipe works.
- **Phase 2** — Accessibility Service wiring: phone can tap/type/open apps
  on command.
- **Phase 3** — PocketClaw (local model) parses natural language into
  actions instead of hardcoded commands.
- **Phase 4** — confirmation step for irreversible actions + audit log.

## Setup (Phase 1)

### 1. Create a Telegram bot
1. Open Telegram, message [@BotFather](https://t.me/BotFather)
2. Send `/newbot`, follow the prompts
3. Save the token it gives you (looks like `123456789:ABCdefGhIJKlmNoPQRstuVwxyz`)
4. Message [@userinfobot](https://t.me/userinfobot) to get your own numeric Telegram user ID

### 2. Set up Termux on the phone
1. Install Termux from F-Droid (not Play Store — the Play Store build is outdated)
   https://f-droid.org/en/packages/com.termux/
2. Open Termux and run:
   ```bash
   pkg update && pkg upgrade -y
   pkg install python git -y
   ```

### 3. Get this repo onto the phone
```bash
git clone https://github.com/YOUR_USERNAME/pocket-jarvis.git
cd pocket-jarvis
pip install -r requirements.txt
```

### 4. Configure secrets
```bash
cp .env.example .env
nano .env   # fill in your bot token and user ID
```

### 5. Run it
```bash
python bot.py
```

Send your bot a message on Telegram. It should echo it back within a few
seconds. That's Phase 1 done — the pipe works.

### 6. Keep it running in the background
- Install **Termux:Boot** (also from F-Droid) so scripts survive reboots
- Disable battery optimization for Termux: Android Settings → Apps →
  Termux → Battery → Unrestricted (otherwise Android will kill the
  background process)
- See `docs/keep-alive.md` for the boot script

## Security notes

- **Never commit your `.env` file** — it contains your bot token, which is
  effectively a password. `.gitignore` already excludes it.
- The bot only responds to the Telegram user ID you configure — anyone
  else messaging it gets ignored.
- Nothing in Phase 1 can control the phone or access files — it only
  echoes text. Control capability arrives in Phase 2, deliberately gated
  behind its own permission checks.
