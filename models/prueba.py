import pandas as pd
from shareplum import Office365
from shareplum.site import Version



site_url = 
username = 
password = 
authcookie = Office365(site_url).GetCookies(username=username, password=password)
p_ctx = Office365(site_url, authcookie=authcookie)

# authcookie = Office365(site_url, version=Version.v365).GetCookies(username=username, password=password)
# sp_ctx = Office365(site_url, version=Version.v365, authcookie=authcookie)

folder_title = "My Folder"
sp_folder = p_ctx.Web.Folders.GetByTitle(folder_title)

for subfolder in sp_folder.SubFolders:
    print(subfolder.Title)
