from onedrivesdk.helpers import GetAuthCodeServer

# SFTP details
sftp_host = 'sftp.christianbook.com'
sftp_port = 9753
sftp_username = 'davidfowler'
sftp_password = 'd4$j!SKn'
remote_file_path = '/to_david_fowler'
local_file_path = 'AMAZON_REPORT.zip'

# OneDrive details
client_id = '52f2432e-52ef-4bab-adeb-d166447e1b8d'
client_secret = '/5pjQWdiaUu51j@zLSQ7Uw.@TxXMaxbx'
redirect_uri = 'http://localhost:8080'
scopes = ['onedrive.readwrite']

# Download file from SFTP
def download_from_sftp():
  transport = paramiko.Transport((sftp_host, sftp_port))
  transport.connect(username=sftp_username, password=sftp_password)
  sftp = paramiko.SFTPClient.from_transport(transport)
  sftp.get(remote_file_path, local_file_path)
  sftp.close()
  transport.close()

# Unzip the file to a subdirectory
def unzip_file():
  with zipfile.ZipFile(local_file_path, 'r') as zip_ref:
    zip_ref.extractall(Documents)

# Upload file to OneDrive
def upload_to_onedrive():
  client = onedrivesdk.get_default_client(client_id=client_id, scopes=scopes)
  auth_url = client.auth_provider.get_auth_url(redirect_uri)
  code = GetAuthCodeServer.get_auth_code(auth_url, redirect_uri)
  client.auth_provider.authenticate(code, redirect_uri, client_secret)
                                                    
  item = onedrivesdk.Item()
  item.name = 'uploaded_file'
  item.file = onedrivesdk.File()
                                                                    
  with open(local_file_path, 'rb') as file:
    client.item(drive='me', id='root').children[item.name].upload(file)

if __name__ == '__main__':
  download_from_sftp()
  unzip_file()
  upload_to_onedrive()


