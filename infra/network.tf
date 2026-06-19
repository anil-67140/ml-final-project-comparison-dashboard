# Use the account's default VPC/subnet instead of creating a new VPC.
# Creating a custom VPC with NAT gateways is a common way free-tier users
# accidentally rack up charges (NAT Gateway is NOT free). The default VPC
# already has an internet gateway attached at no extra cost.

data "aws_vpc" "default" {
  default = true
}

data "aws_subnets" "default" {
  filter {
    name   = "vpc-id"
    values = [data.aws_vpc.default.id]
  }
}

data "aws_ami" "amazon_linux_2023" {
  most_recent = true
  owners      = ["amazon"]

  filter {
    name   = "name"
    values = ["al2023-ami-*-x86_64"]
  }

  filter {
    name   = "virtualization-type"
    values = ["hvm"]
  }
}
