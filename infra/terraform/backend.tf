terraform {
  backend "s3" {
    bucket         = "<STATE_BUCKET>"
    key            = "<PROJECT>/terraform.tfstate"
    region         = "eu-west-2"
    dynamodb_table = "<STATE_LOCK_TABLE>"
    encrypt        = true
  }
}
