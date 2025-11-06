# aws set up
The lambda reads data from a s3 bucket. Set it up like this.
- `aws s3 mb s3://ucbfcqs # allow public read in AWS UI`
- `aws s3 cp /tmp/fcq.csv s3://ucbfcqs/fcq.csv`


# Versions
- 1.3.0 Added score distribution for an individual instructor
- 1.2.0 Added trend analysis by time going back to 2020
- 1.1.0 Added testing for back end, shared on GitHub