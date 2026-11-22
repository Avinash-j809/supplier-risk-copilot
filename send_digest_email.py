"""
send_digest_email.py — Send the most recent weekly digest via Gmail SMTP.

Reads GMAIL_USER, GMAIL_APP_PASSWORD, DIGEST_RECIPIENT from environment.
If any are missing or SMTP fails, prints a warning and exits 0 so the
workflow continues to the artifact upload step regardless.

Usage:
    python send_digest_email.py
"""

import os
import smtplib
import sys
from email.mime.application import MIMEApplication
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path


def find_latest_digest() -> tuple[Path | None, Path | None]:
    """Return (most-recent .md, most-recent .json) under output/, or (None, None)."""
    output = Path("output")
    if not output.is_dir():
        return None, None
    md_files   = sorted(output.glob("digest_*.md"),   reverse=True)
    json_files = sorted(output.glob("digest_*.json"), reverse=True)
    return (md_files[0] if md_files else None, json_files[0] if json_files else None)


def main() -> None:
    gmail_user   = os.environ.get("GMAIL_USER")
    app_password = os.environ.get("GMAIL_APP_PASSWORD")
    recipient    = os.environ.get("DIGEST_RECIPIENT")

    missing = [k for k, v in {
        "GMAIL_USER": gmail_user,
        "GMAIL_APP_PASSWORD": app_password,
        "DIGEST_RECIPIENT": recipient,
    }.items() if not v]
    if missing:
        print(f"[email] skipping — credentials missing: {', '.join(missing)}")
        sys.exit(0)

    md_path, json_path = find_latest_digest()
    if not md_path:
        print("[email] skipping — no digest files found in output/")
        sys.exit(0)

    md_text  = md_path.read_text(encoding="utf-8")
    date_str = md_path.stem.replace("digest_", "")  # "YYYY-MM-DD"

    # Build multipart message: plain-text body + two file attachments
    msg = MIMEMultipart()
    msg["From"]    = gmail_user
    msg["To"]      = recipient
    msg["Subject"] = f"Supplier Risk Watch List — {date_str}"
    msg.attach(MIMEText(md_text, "plain", "utf-8"))

    for path in [md_path, json_path]:
        if path and path.exists():
            part = MIMEApplication(path.read_bytes(), Name=path.name)
            part["Content-Disposition"] = f'attachment; filename="{path.name}"'
            msg.attach(part)

    try:
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.ehlo()
            server.starttls()
            server.login(gmail_user, app_password)
            server.sendmail(gmail_user, recipient, msg.as_string())
        print(f"[email] sent to {recipient}")
    except smtplib.SMTPException as exc:
        print(f"[email] SMTP error — {type(exc).__name__}: {exc}")
        sys.exit(0)
    except Exception as exc:
        print(f"[email] unexpected error — {type(exc).__name__}: {exc}")
        sys.exit(0)


if __name__ == "__main__":
    main()
