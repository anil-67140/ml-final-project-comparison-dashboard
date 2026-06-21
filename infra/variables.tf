variable "aws_region" {
  description = "AWS region to deploy into. Pick the region closest to you to reduce latency; free tier applies in all regions."
  type        = string
  default     = "us-east-1"
}

variable "instance_type" {
  description = "EC2 instance type. t3.micro is the current (2024+) free-tier-eligible type for new accounts (replaces the older t2.micro on most accounts created after 2021). Check your AWS Billing > Free Tier page to confirm which one your account gets for free — some older accounts still only get t2.micro free."
  type        = string
  default     = "t3.micro"
}

variable "root_volume_gb" {
  description = "Root EBS volume size in GB. Free tier includes 30GB of gp2/gp3 EBS storage per month. Kept small here since the repo + venv + models are lightweight."
  type        = number
  default     = 30
}

variable "key_pair_name" {
  description = "Name of an existing EC2 key pair (created in AWS Console > EC2 > Key Pairs) used for SSH access. You must create this yourself first and download the .pem file."
  type        = string
}

variable "ssh_allowed_cidr" {
  description = "CIDR block allowed to SSH into the instance. Set this to YOUR_IP/32, not 0.0.0.0/0, to avoid exposing SSH to the whole internet. Find your IP at https://checkip.amazonaws.com"
  type        = string
}

variable "github_repo_url" {
  description = "HTTPS URL of your GitHub repo to clone onto the instance."
  type        = string
  default     = "https://github.com/anil-67140/ml-final-project-comparison-dashboard.git"
}

variable "app_choice" {
  description = "Which app(s) to run on boot: 'flask', 'streamlit', or 'both'. Running 'both' is fine on t3.micro since neither is compute-heavy with pre-trained models, but uses a bit more of the 1GB RAM."
  type        = string
  default     = "both"

  validation {
    condition     = contains(["flask", "streamlit", "both"], var.app_choice)
    error_message = "app_choice must be one of: flask, streamlit, both."
  }
}
