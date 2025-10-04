# aws set up
The lambda reads data from a s3 bucket. Set it up like this.
- `aws s3 mb s3://ucbfcqs # allow public read in AWS UI`
- `aws s3 cp /tmp/fcq.csv s3://ucbfcqs/fcq.csv`