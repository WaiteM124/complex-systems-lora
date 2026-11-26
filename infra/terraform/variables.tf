variable "project" {
  type        = string
  description = "Short project name used in resource names and default tags."
}

variable "owner" {
  type        = string
  description = "Owner tag, for example a GitHub handle."
}

variable "environment" {
  type        = string
  description = "Deployment environment tag (for example local, staging, or production)."
}

variable "aws_region" {
  type        = string
  description = "AWS region for this stack."
  default     = "eu-west-2"
}

variable "budget_limit_usd" {
  type        = string
  description = "Monthly cost budget in USD."
  default     = "20"
}

variable "budget_alert_email" {
  type        = string
  description = "Email that receives the budget alarm. Do not commit a personal address in examples."
}
