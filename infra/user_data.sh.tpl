#!/bin/bash
set -euxo pipefail

# Log everything to a file so you can debug via SSH if something fails:
# cat /var/log/user-data.log
exec > /var/log/user-data.log 2>&1

dnf update -y
dnf install -y python3.11 python3.11-pip git

# App lives here
APP_DIR="/opt/ml-dashboard"
rm -rf "$APP_DIR"
git clone "${github_repo_url}" "$APP_DIR"
cd "$APP_DIR"

python3.11 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# Models are already trained per the user's repo - train_models.py is run
# once if the models/ folder is missing, so a reboot doesn't retrain
# (saves CPU credits on t3.micro, which is burstable and limited).
if [ ! -f "$APP_DIR/models/metrics.json" ]; then
  python train_models.py
fi

deactivate

# --- systemd service: Flask ---
cat > /etc/systemd/system/flask-dashboard.service <<EOF
[Unit]
Description=Flask ML Dashboard
After=network.target

[Service]
WorkingDirectory=$APP_DIR
ExecStart=$APP_DIR/venv/bin/python $APP_DIR/app.py
Restart=on-failure
User=ec2-user
Environment=FLASK_RUN_HOST=0.0.0.0

[Install]
WantedBy=multi-user.target
EOF

# --- systemd service: Streamlit ---
cat > /etc/systemd/system/streamlit-dashboard.service <<EOF
[Unit]
Description=Streamlit ML Dashboard
After=network.target

[Service]
WorkingDirectory=$APP_DIR
ExecStart=$APP_DIR/venv/bin/streamlit run $APP_DIR/dashboard.py --server.address=0.0.0.0 --server.port=8501
Restart=on-failure
User=ec2-user

[Install]
WantedBy=multi-user.target
EOF

chown -R ec2-user:ec2-user "$APP_DIR"

systemctl daemon-reload

%{ if app_choice == "flask" || app_choice == "both" ~}
systemctl enable --now flask-dashboard.service
%{ endif ~}

%{ if app_choice == "streamlit" || app_choice == "both" ~}
systemctl enable --now streamlit-dashboard.service
%{ endif ~}

echo "Bootstrap complete."
