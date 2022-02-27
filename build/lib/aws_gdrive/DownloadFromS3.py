import boto3
import logging
from tqdm import tqdm
import os
from botocore.exceptions import ClientError
from boto.s3.connection import S3Connection


def splitall(path):
    allparts = []
    while 1:
        parts = os.path.split(path)
        if parts[0] == path:  # sentinel for absolute paths
            allparts.insert(0, parts[0])
            break
        elif parts[1] == path:  # sentinel for relative paths
            allparts.insert(0, parts[1])
            break
        else:
            path = parts[0]
            allparts.insert(0, parts[1])
    return allparts


def hook(t):
    def inner(bytes_amount):
        t.update(bytes_amount)

    return inner


def S3_download(file_name, save_file_name, bucket='bonfire-data-team-share', show_progress=True):
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


def upload_file(
    file_name, save_file_name=None, bucket='bonfire-data-team-share', show_progress=True
):
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
            with tqdm(total=float(os.path.getsize(file_name)), unit='B', unit_scale=True,
                      desc=file_name) as t:
                s3_client.upload_file(file_name, bucket, save_file_name, Callback=hook(t))
        else:
            s3_client.upload_file(file_name, bucket, save_file_name)
    except ClientError as e:
        logging.error(e)
        return False
    return True


def S3_download_folder(credentials_dict, folder_name, bucket_name, save_path=None):
    """
    This function is able to download a single file or a whole folder, but it requires the credentials

    @param credentials_dict: {'Access key ID': your Access key ID, 'Secret access key': your secret access key}
    @param folder_name: your remote folder name or file name, need to be this format: folder/sub
    @param bucket_name: bucket_name is a string or your bucket name
    @param save_path: save path is where you want to save the folder or file, default is your current working directory

    """
    if save_path:
        pass
    else:
        save_path = os.getcwd()
    conn = S3Connection(credentials_dict['Access key ID'], credentials_dict['Secret access key'])
    bucket = conn.get_bucket(bucket_name)
    folder_content = []
    for key in bucket.list():
        diretory_name = key.name
        if diretory_name.startswith(folder_name):
            # name_parse=diretory_name.split('/')
            name_parse = splitall(diretory_name)
            if not name_parse[-1]:
                try:
                    os.mkdir(os.path.join(save_path, diretory_name))
                except:
                    pass
            else:
                for i in range(1, len(name_parse)):
                    try:
                        os.mkdir(os.path.join(save_path, *name_parse[:i]))
                    except:
                        pass
                folder_content.append(diretory_name)
    s3 = boto3.resource('s3')
    s31 = boto3.client('s3')
    for file_name in folder_content:
        file_object = s3.Object(bucket_name, file_name)
        filesize = file_object.content_length
        with tqdm(total=filesize, unit='B', unit_scale=True, desc=file_name) as t:
            s31.download_file(
                bucket_name, file_name, os.path.join(save_path, file_name), Callback=hook(t)
            )


def S3_upload_folder(
    foler_name, remote_folder, bucket, ignore_files=['.DS_Store'], show_progress=True
):
    """
    This function is able to upload a folder or a file to specific place on S3
    @param foler_name: need to be like this format: /Users/xinzhang/folder/sub
                       or a file like : /Users/xinzhang/folder/sub/data.csv
    @param remote_folder: like this on S3 root/folder/sub
    @param bucket: your bucket, string
    @param ignore_files: a list of string
    @param show_progress: Boolean
    """
    if list(os.walk(foler_name)):
        actuall_folder_name = splitall(foler_name)[-1]
        pre_path = os.path.join(*splitall(foler_name)[:-1])
        for path, subdirs, files in os.walk(foler_name):
            for name in files:
                if name in ignore_files:
                    continue
                full_path = os.path.join(path, name)
                folders = []
                while os.path.split(full_path)[1] != actuall_folder_name:
                    folders.append(os.path.split(full_path)[1])
                    full_path = os.path.split(full_path)[0]
                folders.append(actuall_folder_name)
                folders.reverse()
                file_path = os.path.join(*folders)
                aws_path = os.path.join(remote_folder, file_path)
                upload_file(
                    os.path.join(pre_path, file_path),
                    save_file_name=aws_path,
                    bucket=bucket,
                    show_progress=show_progress
                )
    else:
        file_name = splitall(foler_name)[-1]
        aws_path = os.path.join(remote_folder, file_name)
        upload_file(foler_name, save_file_name=aws_path, bucket=bucket, show_progress=show_progress)
