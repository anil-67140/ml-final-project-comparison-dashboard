output "instance_id" {
  description = "EC2 instance ID"
  value       = aws_instance.ml_dashboard.id
}

output "public_ip" {
  description = "Static public IP (Elastic IP) of the instance"
  value       = aws_eip.ml_dashboard_eip.public_ip
}

output "flask_url" {
  description = "URL for the Flask app once boot script finishes (~2-3 min after apply)"
  value       = "http://${aws_eip.ml_dashboard_eip.public_ip}:5000"
}

output "streamlit_url" {
  description = "URL for the Streamlit dashboard once boot script finishes (~2-3 min after apply)"
  value       = "http://${aws_eip.ml_dashboard_eip.public_ip}:8501"
}

output "ssh_command" {
  description = "Command to SSH into the instance for debugging"
  value       = "ssh -i /path/to/${var.key_pair_name}.pem ec2-user@${aws_eip.ml_dashboard_eip.public_ip}"
}
