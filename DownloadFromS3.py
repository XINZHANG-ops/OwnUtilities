import boto3
from tqdm import tqdm
# Let's use Amazon S3

def hook(t):
  def inner(bytes_amount):
    t.update(bytes_amount)
  return inner

def S3_download(file_name,save_file_name,bucket,show_progress=True):
    if show_progress:
        s3 = boto3.resource('s3')
        s31 = boto3.client('s3')
        file_object = s3.Object(bucket, file_name)
        filesize = file_object.content_length
        with tqdm(total=filesize, unit='B', unit_scale=True, desc=file_name) as t:
            s31.download_file(bucket, file_name, save_file_name, Callback=hook(t))
    else:
        s3 = boto3.client('s3')
        s3.download_file(bucket, file_name, save_file_name)




