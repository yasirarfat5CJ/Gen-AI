import json
import boto3


prompt_data="""
Act as a shakespeare and write a poem on generative AI"""

bedrock=boto3.client(service_name="bedrock-runtime")

payload={
    "prompt":"[INST]"+prompt_data+"[/INST]",
    "max_gen_len":512,
    "temperature":0.5,
    "top_p":0.9
}

body=json.dumps(payload)
model_id="meta.llama3-70b-instruct-v1:0"

res=bedrock.invoke_model(
    body=body,
    modelId=model_id,
    accept="application/json",
    contentType="application/json"
)

res_body=json.loads(res["body"].read())

res_text=res_body['generation']
print(res_text)