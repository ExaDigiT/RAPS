MIT supercloud data. https://dcc.mit.edu/dataconda 

To install S3 client. 
sudo apt  install awscli 

aws s3 ls s3://mit-supercloud-dataset/datacenter-challenge/202201/ --no-sign-request



# Conda env creation: 
conda create --name parser \
boto3 numpy pandas spyder pyarrow fastparquet h5py matplotlib seaborn scikit-learn scipy requests beautifulsoup4 sqlalchemy openpyxl xlrd 


conda activate parser 

spyder 

From within spyder you can access the data using parse_mit_data.py 


