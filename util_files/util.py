import constants
import os
import io
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from google.colab import auth
from oauth2client.client import GoogleCredentials

def downloadfile(id, path):
    auth.authenticate_user()
    gauth = GoogleAuth()
    gauth.credentials = GoogleCredentials.get_application_default()
    drive = GoogleDrive(gauth)

    os.makedirs(path)
    os.chdir(path)
    # 2. Auto-iterate using the query syntax
    #    https://developers.google.com/drive/v2/web/search-parameters
    file_list = drive.ListFile(
        {'q': "'"+id+"' in parents"}).GetList()  # use your own folder ID here

    for f in file_list:
        # 3. Create & download by id.
        print('title: %s, id: %s' % (f['title'], f['id']))
        fname = f['title']
        print('downloading to {}'.format(fname))
        f_ = drive.CreateFile({'id': f['id']})
        f_.GetContentFile(fname)


def download_model(model):
    path = '../model/'+model
    download_ids = get_download_ids(model)
    if os.path.exists(path):
        return
    else:

        for id in download_ids:
            downloadfile(id,path)

def get_download_ids(model):
    model_ids=model+'_download_ids'
    return constants.model_ids