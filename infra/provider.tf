provider "aws" {
  region = var.aws_region

  default_tags {
    tags = {
      Project   = "ml-final-project-comparison-dashboard"
      ManagedBy = "terraform"
      Tier      = "free-tier"
    }
  }
}
