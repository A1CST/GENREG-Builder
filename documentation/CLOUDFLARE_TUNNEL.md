# Exposing the Assistant web UI at genreg.online (Cloudflare Tunnel)

This puts the web UI online at **https://genreg.online** with:

- **no inbound ports opened** on your router/firewall,
- your **home IP (71.60.202.164) hidden** behind Cloudflare,
- a **real TLS certificate** (Cloudflare-managed) — so the TOTP code and session
  cookie are encrypted end-to-end,
- automatic survival of a **dynamic residential IP** (the tunnel dials out).

The app itself keeps binding `127.0.0.1:5008` — Cloudflare's `cloudflared`
daemon connects to it locally and relays public traffic.

> Run the app in proxy mode whenever it's behind the tunnel:
> `python main.py --flask --behind-proxy`
> (Secure HTTPS-only cookie, honours `X-Forwarded-*`, and the localhost
> `/enroll` bypass is disabled — enroll your authenticator from the CLI instead.)

---

## 0. One-time prerequisites

1. **Add genreg.online to Cloudflare** (free plan is fine): create a Cloudflare
   account → "Add a site" → `genreg.online`. Cloudflare gives you two
   nameservers.
2. **Point the domain's nameservers at Cloudflare** at your registrar (where you
   bought genreg.online). Wait for it to go "Active" in Cloudflare (minutes to a
   few hours).

You do **not** create an A record to your IP — the tunnel makes a CNAME for you.

---

## 1. Install cloudflared (Linux / x86_64)

```bash
# Debian/Ubuntu .deb (recommended — gets you the service installer too)
curl -L https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb -o /tmp/cloudflared.deb
sudo dpkg -i /tmp/cloudflared.deb
cloudflared --version
```

(Other platforms: https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/downloads/)

## 2. Authenticate cloudflared to your Cloudflare account

```bash
cloudflared tunnel login
```

This opens a browser; pick the **genreg.online** zone. It writes a cert to
`~/.cloudflared/cert.pem`.

## 3. Create the tunnel

```bash
cloudflared tunnel create genreg
```

This prints a **Tunnel ID** (a UUID) and writes credentials to
`~/.cloudflared/<TUNNEL_ID>.json`.

## 4. Write the tunnel config

Create `~/.cloudflared/config.yml` (replace `<TUNNEL_ID>` and the home dir):

```yaml
tunnel: <TUNNEL_ID>
credentials-file: /home/payton-millnet/.cloudflared/<TUNNEL_ID>.json

ingress:
  - hostname: genreg.online
    service: http://127.0.0.1:5008
  - service: http_status:404      # everything else gets a 404
```

## 5. Route the DNS name to the tunnel

```bash
cloudflared tunnel route dns genreg genreg.online
```

This creates the proxied CNAME in Cloudflare DNS automatically.

## 6. Enroll your authenticator (from the machine, before going public)

Because `--behind-proxy` disables the localhost `/enroll` shortcut, set up TOTP
from a terminal — the QR prints right in the terminal, so the secret never
crosses the network:

```bash
python -m genreg_gui.web.auth totp
```

Scan the terminal QR (or type the manual key) into your authenticator app.

## 7. Run it

Two processes (use two terminals, or install both as services):

```bash
# terminal A — the app in proxy mode
python main.py --flask --behind-proxy

# terminal B — the tunnel
cloudflared tunnel run genreg
```

Visit **https://genreg.online** → TOTP login → you're in.

### Optional: run on boot as services

```bash
# tunnel as a system service
sudo cloudflared service install
sudo systemctl enable --now cloudflared
```

(Run the app via your own systemd unit or autostart; keep `--behind-proxy`.)

---

## Hardening you get for free with this setup

- **Zero open inbound ports** and a **hidden origin IP** — the attack surface is
  Cloudflare's edge, not your home router.
- **TLS everywhere** — the session cookie is marked `Secure`, so it's only ever
  sent over HTTPS.

## Strongly recommended extra: Cloudflare Access

Put a second gate *in front of* the app at Cloudflare's edge so unauthenticated
traffic never even reaches your machine:

- Cloudflare dashboard → **Zero Trust → Access → Applications** → add a
  self-hosted app for `genreg.online`, policy = "allow only your email"
  (one-time PIN or Google login).

This means a stranger hitting genreg.online is stopped at Cloudflare and never
touches the Flask app at all — defence in depth on top of the TOTP login. For a
tool that controls your desktop, this is worth the five minutes.
