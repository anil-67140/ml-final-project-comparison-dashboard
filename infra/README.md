# Terraform: ml-final-project-comparison-dashboard on AWS EC2 (Free Tier)

Deploys your Flask + Streamlit dashboard to a single **t3.micro** EC2 instance.

## Why EC2 + t3.micro (and not ECS/Lambda/Elastic Beanstalk)
- t3.micro: 750 free hours/month for 12 months on new accounts — enough for
  ONE instance running 24/7 all month.
- No load balancer, no NAT gateway, no RDS — these are the most common ways
  people accidentally burn free-tier credit. This setup has none of them.
- Uses your account's **default VPC** (already exists, no extra charge)
  instead of creating a new one.
- EBS root volume set to 8GB, well under the 30GB free allowance.

## Files
| File | Purpose |
|---|---|
| `versions.tf` | Terraform + AWS provider version pins |
| `provider.tf` | AWS provider config + default tags |
| `variables.tf` | All configurable inputs |
| `network.tf` | Reads default VPC/subnet, finds latest Amazon Linux 2023 AMI |
| `security_group.tf` | Firewall rules — SSH locked to your IP, app ports open |
| `main.tf` | The EC2 instance + Elastic IP |
| `user_data.sh.tpl` | Boot script: installs Python, clones repo, runs both apps as systemd services |
| `outputs.tf` | Prints URLs/IP after `apply` |
| `terraform.tfvars.example` | Template for your personal values — copy to `terraform.tfvars` |
| `.gitignore` | Keeps state files, `.tfvars`, and `.pem` keys out of git |

## Setup steps

### 1. Prerequisites (one-time, in AWS Console)
1. Create an EC2 **key pair**: EC2 Console → Key Pairs → Create → RSA → `.pem` format.
   Download the `.pem` file, then `chmod 400 your-key.pem`.
2. Find your IP: visit https://checkip.amazonaws.com — note it down.

### 2. Configure
```bash
cp terraform.tfvars.example terraform.tfvars
# edit terraform.tfvars: set key_pair_name and ssh_allowed_cidr (YOUR_IP/32)
```

### 3. Deploy
```bash
terraform init
terraform plan    # review what will be created
terraform apply   # type "yes" to confirm
```

Wait ~2-3 minutes after apply completes for the boot script to finish
installing dependencies and starting the apps.

### 4. Access
Terraform will print:
- `flask_url` → `http://<ip>:5000`
- `streamlit_url` → `http://<ip>:8501`
- `ssh_command` → for debugging

### 5. Debugging if a page doesn't load
```bash
ssh -i your-key.pem ec2-user@<public_ip>
sudo cat /var/log/user-data.log        # full boot script log
sudo systemctl status flask-dashboard
sudo systemctl status streamlit-dashboard
sudo journalctl -u streamlit-dashboard -f   # live logs
```

### 6. Tear down (IMPORTANT — do this when done demoing)
```bash
terraform destroy
```
This deletes the instance AND releases the Elastic IP. Leaving the instance
running 24/7 is fine within free tier for the first 12 months, but always
`destroy` once you no longer need it to avoid any risk of charges after
free tier expires or limits are hit.

## Free-tier guardrails built in
- Single instance only — free tier covers one t3.micro running full-time,
  not multiple.
- No load balancer, NAT gateway, or RDS instance created.
- EBS volume well under the 30GB free allowance.
- Models are NOT retrained on every reboot — `train_models.py` only runs
  if `models/metrics.json` doesn't already exist, since your repo says
  models are pre-trained. This avoids wasting CPU credits on a burstable
  instance.
- SSH restricted to your IP only, not open to the whole internet.

## Things to watch yourself
- **Don't leave the Elastic IP unattached.** `terraform destroy` releases
  it automatically — just don't manually detach it without releasing it.
- **Don't create a second t3.micro elsewhere** in the same account/region
  while this one runs — free tier hours are pooled across the account.
- After 12 months (or if you're on an account past its free-tier window),
  t3.micro is roughly $0.0104/hr (us-east-1) — check current AWS pricing.
