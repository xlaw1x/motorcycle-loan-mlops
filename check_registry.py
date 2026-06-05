import mlflow
from mlflow import MlflowClient

mlflow.set_tracking_uri("sqlite:///mlflow.db")
client = MlflowClient()

found = False
for rm in client.search_registered_models():
    for mv in client.search_model_versions(f"name='{rm.name}'"):
        print(f"  {mv.name}  v{mv.version}  ->  stage={mv.current_stage}  aliases={list(mv.aliases) if hasattr(mv,'aliases') else 'n/a'}")
        found = True

if not found:
    print("NONE REGISTERED")