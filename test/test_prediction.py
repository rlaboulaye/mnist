import sys
import requests
import numpy as np
import json

url = 'http://ec2-35-162-99-16.us-west-2.compute.amazonaws.com:5000/api/mnist/prediction'
r = requests.post(url, files={'image': open(sys.argv[1],'rb')})
results = json.loads(r.text)['results']
print(results)
print(np.argmax(results))
