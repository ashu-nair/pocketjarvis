# Phase 2: Giving the phone actual control

## Why this is a separate app, not just more Termux code

Termux runs as a normal, unprivileged Android app. It cannot tap buttons in
WhatsApp, type into other apps, or read what's on screen — that capability
is gated behind Android's **Accessibility Service** API, and only an app
that has been granted the Accessibility permission can use it.

So Phase 2 needs a real Android app (even a tiny one) that:
1. Requests the Accessibility permission
2. Exposes a way for `bot.py` (running in Termux) to tell it what to do

`bot.py` and this new app talk to each other over `http://127.0.0.1` —
same device, no network exposure, nothing leaves the phone.

## Path A — No-code, fast (recommended to get moving today)

Use **MacroDroid** (free tier is plenty). It already has Accessibility
permission wired up internally, and it supports an **HTTP request trigger**
— meaning it can run a macro (open an app, type text, tap a button) when
it receives a local web request. `bot.py` just sends that request.

You get real device control working *today* without touching Android Studio.
See `docs/macrodroid-setup.md`.

**Trade-off**: you're bounded by what MacroDroid's macro actions support.
Fine for "open app / type text / send message" — more limiting if you
later want fine-grained custom logic (e.g., reading screen content back
into the agent's reasoning).

## Path B — Custom Accessibility Service (the long-term foundation)

A minimal Kotlin Android app that:
- Implements `AccessibilityService` directly — full read + control access
- Runs a tiny local HTTP server so `bot.py` can send it commands
- You own 100% of the logic, extendable forever (this is what Droidrun /
  OpenGUI-style frameworks do under the hood)

**Trade-off**: needs Android Studio to build and install (APK signing,
etc.) — more setup, but this is genuinely the foundation for everything in
the "where this can grow" list from earlier (voice interface, memory,
proactive behavior — all of it eventually routes through this layer).

Source is in `android-accessibility-app/`. You don't need to build this
today — Path A gets you moving now, and you can swap to Path B whenever
you're ready for full custom control. `actions.py` is written so swapping
the backend later doesn't touch `bot.py` at all.

## Recommended order

1. Do Path A now (MacroDroid) — get "open app" and "type text" actually
   working end-to-end today, commands sent from Telegram.
2. Once that feels solid, build Path B when you have time to sit with
   Android Studio for an afternoon — it's a straight upgrade, same
   interface from `bot.py`'s point of view.
