# aws-ml-cpu-v-gpu
A lab to compare CPU to GPU performance using the AWS Deep Learning AMI and p2.xlarge instance type.

![Lab environment](https://user-images.githubusercontent.com/3911650/33036125-c92a5c9a-cdea-11e7-8563-226d5d2c20f4.png)

## Getting Started
Deploy the CloudFormation stack in the template in `infrastructure/`. The template creates a user with the following credentials and minimal required permisisons to complete the Lab:
- Username: _student_
- Password: _password_

## Instructions
- Connect to the instance using the SSH username: _ubuntu_. 
- Run the Jupyter notebook server that comes pre-installed on the [Amazon Deep Learning AMI](https://aws.amazon.com/marketplace/pp/B01M0AXXQB): `jupyter notebook` 
- SSH tunnel to the notebook server running on port 8888
- Open a browser to the notebook server on localhost. Get the URL with token from the command `jupyter notebook list`
- Create a new notebook using the python 3.6 and TensorFlow environment
- Paste the code in `src/tf_matmul.py` into a cell
- Run the notebook and analyze the result charts

## Cleaning Up
Delete the CloudFormation stack to remove all the resources. No resources are created outside of those created by the template.