"""
****************************************
 * @author: Xin Zhang
 * Date: 5/28/21
****************************************
"""
import requests
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
import os
from tqdm import tqdm


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


def upload_folder_to_gdrive(
    local_folder_path, remote_folder_id, file_suffix=None, exclude_exist=True
):
    """

    @param local_folder_path:
    @param remote_folder_id:
    @param file_suffix: use to select one file type under folder, for example, .json, .txt
    @param exclude_exist:
    @return:
    """
    gauth = GoogleAuth()
    drive = GoogleDrive(gauth)
    alredy_exist = []
    file_list = drive.ListFile({
        'q': "'{}' in parents and trashed=false".format(remote_folder_id)
    }).GetList()
    for file in file_list:
        alredy_exist.append(file['title'])
        #print('title: %s, id: %s' % (file['title'], file['id']))

    all_files = []
    for dirpath, dirnames, filenames in os.walk(local_folder_path):
        if file_suffix is None:
            pass
        else:
            filenames = [f for f in filenames if f.endswith(file_suffix)]
        for filename in filenames:
            all_files.append(os.path.join(dirpath, filename))
    print('Total files: ', len(all_files))
    if exclude_exist:
        all_files = [i for i in all_files if i not in alredy_exist]
    print('Need to be uploaded: ', len(all_files))
    for upload_file in tqdm(all_files):
        gfile = drive.CreateFile({'parents': [{'id': remote_folder_id}]})
        # Read file and set it as the content of this instance.
        gfile.SetContentFile(upload_file)
        gfile.Upload()


def demo_download():
    import os
    # make sure the file is public (anyone can access with the link)
    # get fild id from sharable link of file
    file_id = '12n3iL4HCsHRzqYZz_dAD7YHqxfbDfRq-'
    destination = os.getcwd()
    download_file_from_google_drive(file_id, destination + '/TrainDataCurrent.zip')


def demo_upload():
    """
    First make sure the client_secret.json is under your working directory
    follow this tutorial to get client_secret.json
    https://pythonhosted.org/PyDrive/quickstart.html

    if upload has http error, try run below:

    pip install httplib2==0.15.0
    pip install google-api-python-client==1.6

    @return:
    """
    # link to this folder
    # https://drive.google.com/drive/u/1/folders/119aVANIJT9eC32-d4azwBaFIASqsTYyY

    folder_id = '119aVANIJT9eC32-d4azwBaFIASqsTYyY'
    fold_path = '/home/xinzhang/Downloads'
    upload_folder_to_gdrive(fold_path, folder_id, file_suffix='txt', exclude_exist=True)
