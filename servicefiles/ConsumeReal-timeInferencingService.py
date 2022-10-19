from azureml.core.webservice import AksWebservice

ws=azureml.core.workspace.from_config()
#Consume a real-time inferencing service
#Use a REST endpoint
# Get the deployed service
service = AksWebservice(name='classifier-service', workspace=ws)

endpoint = service.scoring_uri
print(endpoint)


import requests
import json

# An array of new data cases
x_new = [[0.1,2.3,4.1,2.0],
         [0.2,1.8,3.9,2.1]]

# Convert the array to a serializable list in a JSON document
json_data = json.dumps({"data": x_new})
primary_key, secondary_key = service.get_keys()
token=service.get_token()
# Set the content type in the request headers
request_headers = { 'Content-Type':'application/json',
                    "Authorization":"Bearer " + token} #key_or_token }

# Call the service
response = requests.post(url = endpoint,
                         data = json_data,
                         headers = request_headers)

# Get the predictions from the JSON response
predictions = json.loads(response.json())

# Print the predicted class for each case.
for i in range(len(x_new)):
    print ((x_new[i]), predictions[i] )