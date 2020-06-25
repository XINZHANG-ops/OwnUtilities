import boto3
import logging
from tqdm import tqdm
import os
from botocore.exceptions import ClientError

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



def upload_file(bucket,file_name,save_file_name=None,show_progress=True):
    """Upload a file to an S3 bucket

    :param file_name: File to upload
    :param bucket: Bucket to upload to
    :param object_name: S3 object name. If not specified then file_name is used
    :return: True if file was uploaded, else False
    """
    # If S3 object_name was not specified, use file_name
    if save_file_name is None:
        save_file_name = file_name

    # Upload the file
    s3_client = boto3.client('s3')
    try:
        if show_progress:
            with tqdm(total=float(os.path.getsize(file_name)), unit='B', unit_scale=True, desc=file_name) as t:
                s3_client.upload_file(file_name, bucket,save_file_name,Callback=hook(t))
        else:
            s3_client.upload_file(file_name, bucket, save_file_name)
    except ClientError as e:
        logging.error(e)
        return False
    return True
