'''

Client to test the server

'''

import requests

url = 'http://127.0.0.1:5000/sendJsonData'
# headers = {'Authorization' : ‘(some auth code)’, 'Accept' : 'application/json', 'Content-Type' : 'application/json'}
r = requests.post(url, data=open('COMMUNICATE_3_DOKE.json', 'rb'), headers=None)