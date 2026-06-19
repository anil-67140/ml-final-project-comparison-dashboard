# The EC2 instance running both apps as systemd services.
# t3.micro = 2 vCPU (burstable), 1GB RAM - free tier covers 750 hrs/month,
# which is enough for ONE instance running 24/7 for a 30-day month.
# IMPORTANT: free tier only covers ONE running instance at a time across
# your whole account. Don't spin up a second t3.micro elsewhere while this
# one is running, or you'll start paying for the extra hours.

resource "aws_instance" "ml_dashboard" {
  ami                    = data.aws_ami.amazon_linux_2023.id
  instance_type          = var.instance_type
  subnet_id              = data.aws_subnets.default.ids[0]
  vpc_security_group_ids = [aws_security_group.app_sg.id]
  key_name               = var.key_pair_name

  # gp3 is the current default and cheaper per-GB than gp2, and still
  # covered by the same 30GB-free EBS allowance.
  root_block_device {
    volume_type = "gp3"
    volume_size = var.root_volume_gb
    encrypted   = true
  }

  # Avoids accidental charges from metadata service misuse; IMDSv2 only.
  metadata_options {
    http_tokens   = "required"
    http_endpoint = "enabled"
  }

  user_data = templatefile("${path.module}/user_data.sh.tpl", {
    github_repo_url = var.github_repo_url
    app_choice      = var.app_choice
  })

  # Forces a clean redeploy if you change the repo URL or app choice,
  # without having to manually terminate the instance.
  user_data_replace_on_change = true

  tags = {
    Name = "ml-dashboard-instance"
  }
}

# Elastic IP so the public address doesn't change if the instance reboots.
# NOTE: an EIP is free ONLY while it's attached to a running instance.
# If you stop the instance, either terminate the EIP too or keep the
# instance running - an EIP sitting unattached is billed hourly.
resource "aws_eip" "ml_dashboard_eip" {
  instance = aws_instance.ml_dashboard.id
  domain   = "vpc"

  tags = {
    Name = "ml-dashboard-eip"
  }
}
