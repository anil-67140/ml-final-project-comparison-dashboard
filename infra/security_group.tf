resource "aws_security_group" "app_sg" {
  name        = "ml-dashboard-sg"
  description = "Allow SSH (restricted), Flask (5000), and Streamlit (8501)"
  vpc_id      = data.aws_vpc.default.id

  ingress {
    description = "SSH - restricted to your IP only"
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = [var.ssh_allowed_cidr]
  }

  ingress {
    description = "Flask app"
    from_port   = 5000
    to_port     = 5000
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  ingress {
    description = "Streamlit dashboard"
    from_port   = 8501
    to_port     = 8501
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    description = "Allow all outbound (needed for pip install, git clone, apt)"
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name = "ml-dashboard-sg"
  }
}
