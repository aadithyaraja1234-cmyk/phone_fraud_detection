# Deploying to PythonAnywhere (free, always-on)

PythonAnywhere's free "Beginner" tier hosts one web app that stays running
all the time, no credit card required. Outbound access is limited to a
whitelist, but it includes GitHub and PyPI, so `git clone` and `pip
install` both work.

## 1. Create an account
Sign up at https://www.pythonanywhere.com/registration/register/beginner/

## 2. Clone the repo
Open a **Bash console** from the PythonAnywhere dashboard (`Consoles` tab
→ `Bash`) and run:

```bash
git clone https://github.com/aadithyaraja1234-cmyk/phone_fraud_detection.git
```

## 3. Create a virtualenv and install dependencies

```bash
mkvirtualenv --python=/usr/bin/python3.10 fraud-env
cd phone_fraud_detection
pip install -r requirements.txt
```

(Use whichever Python 3.x is available on your account — check with
`ls /usr/bin/python3.*`. Any 3.9+ works; the app doesn't need 3.13.)

## 4. Create the web app
Go to the **Web** tab → **Add a new web app** → choose **Manual
configuration** (not "Flask", since the app already exists) → pick the
same Python version as your virtualenv.

Set these fields on the Web tab:
- **Source code**: `/home/<your-username>/phone_fraud_detection`
- **Working directory**: `/home/<your-username>/phone_fraud_detection`
- **Virtualenv**: `/home/<your-username>/.virtualenvs/fraud-env`

## 5. Edit the WSGI configuration file
Click the WSGI configuration file link on the Web tab (something like
`/var/www/<your-username>_pythonanywhere_com_wsgi.py`) and replace its
contents with:

```python
import sys
import os

path = "/home/<your-username>/phone_fraud_detection"
if path not in sys.path:
    sys.path.insert(0, path)

os.chdir(path)

from app import app as application
```

Replace `<your-username>` with your actual PythonAnywhere username in
both this file and the Web tab paths above.

## 6. Reload
Click the green **Reload** button on the Web tab. Your app will then be
live at `https://<your-username>.pythonanywhere.com` — always on, no
sleep/cold-start.

## Updating after future code changes
```bash
cd ~/phone_fraud_detection
git pull
workon fraud-env
pip install -r requirements.txt   # only needed if requirements changed
```
Then hit **Reload** on the Web tab again.

## Notes
- Free accounts get limited daily CPU seconds; this app is lightweight
  (single RandomForest inference per upload) so normal use is well within
  limits.
- `FLASK_DEBUG` must stay unset/`0` in production — the WSGI setup above
  doesn't call `app.run()` at all (PythonAnywhere's own server handles
  that), so debug mode never applies here regardless.
