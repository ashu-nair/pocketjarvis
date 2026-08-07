# MacroDroid setup (Path A)

Gets `bot.py` able to trigger real actions on the phone — open an app,
type text — using MacroDroid as the execution layer.

## 1. Install MacroDroid

From the Play Store: search "MacroDroid". Free tier allows 5 macros,
which is plenty to start.

On first launch, grant it:
- Accessibility permission (Settings → Accessibility → MacroDroid → On)
- Display over other apps (needed for some actions)
- Ignore battery optimization (same reasoning as Termux — don't let
  Android kill it)

## 2. Create your first macro: "Open App"

1. Open MacroDroid → tap **+** to add a macro
2. **Trigger** → Connectivity → **Webhook / HTTP Request Received**
   - This gives you a local webhook URL, something like:
     `http://127.0.0.1:port/trigger/xxxxxxx`
   - Note the exact URL and port MacroDroid shows you
3. **Action** → Application → **Launch Application**
   - Pick a test app (e.g. Play Store, or any game you want to install later)
4. Save the macro, name it `open_app`

## 3. Create a second macro: "Type Text"

1. New macro → same **Webhook** trigger type (different URL/port)
2. **Action** → Input/Keyboard → **Text Simulate Input** (exact name varies
   by MacroDroid version — look for anything under "Input" that types text
   into the currently focused field)
3. Save, name it `type_text`

You now have two local webhook URLs. Note both down — you'll put them in
`.env` next.

## 4. Wire it into bot.py

Add these to your `.env` (see updated `.env.example`):

```
MACRODROID_OPEN_APP_URL=http://127.0.0.1:PORT/trigger/xxxxxxx
MACRODROID_TYPE_TEXT_URL=http://127.0.0.1:PORT/trigger/yyyyyyy
```

`actions.py` (new file) reads these and exposes simple functions that
`bot.py` calls. Test it by messaging your Telegram bot:

```
/open playstore
/type hello from pocket jarvis
```

(exact command parsing is in the updated `bot.py` — literal commands for
now, natural language comes in Phase 3 with PocketClaw)

## Notes

- MacroDroid webhooks only listen on `127.0.0.1` (localhost) by default —
  this is exactly what you want. Nothing is exposed to the network.
- Each new action you want (open specific app, send WhatsApp message, tap
  a specific button) = one more MacroDroid macro + one more webhook URL.
  It scales fine for a handful of actions; once you're managing 10+, that's
  your signal to move to Path B for real programmatic control.
