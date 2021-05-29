"""
****************************************
 * @author: Xin Zhang
 * Date: 5/28/21
****************************************
"""
import requests


def download_file_from_google_drive(id, save_path):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params={'id': id}, stream=True)
    token = _get_confirm_token(response)

    if token:
        params = {'id': id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)

    _save_response_content(response, save_path)


def _get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None


def _save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)


def demo():
    import os
    # make sure the file is public (anyone can access with the link)
    # get fild id from sharable link of file
    file_id = '12n3iL4HCsHRzqYZz_dAD7YHqxfbDfRq-'
    destination = os.getcwd()
    download_file_from_google_drive(file_id, destination + '/TrainDataCurrent.zip')
