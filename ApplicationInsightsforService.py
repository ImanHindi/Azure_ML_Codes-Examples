#Application Insights



from azureml.core import Workspace

ws = Workspace.from_config()
ws.get_details()['applicationInsights']



#Enable Application Insights for a service
dep_config = AciWebservice.deploy_configuration(cpu_cores = 1,
                                                memory_gb = 1,
                                                enable_app_insights=True)


#If you want to enable Application Insights for a service that is already deployed, you can modify the deployment configuration for Azure Kubernetes Service (AKS) based services in the Azure portal.
#  Alternatively, you can update any web service by using the Azure Machine Learning SDK

service = ws.webservices['my-svc']
service.update(enable_app_insights=True)


#Capture and view telemetry
#Application Insights automatically captures any information written to the standard output and error logs, and provides a query capability to view data in these logs.

#Write log data
def init():
    global model
    model = joblib.load(Model.get_model_path('my_model'))
def run(raw_data):
    data = json.loads(raw_data)['data']
    predictions = model.predict(data)
    log_txt = 'Data:' + str(data) + ' - Predictions:' + str(predictions)
    print(log_txt)
    return predictions.tolist()



#Query logs in Application Insights:
#SQL
#traces
#|where message == "STDOUT"
#  and customDimensions.["Service Name"] = "my-svc"
#| project  timestamp, customDimensions.Content