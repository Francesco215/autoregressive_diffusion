#%%
import boto3
from typing import List
from botocore.exceptions import ClientError
import json
from streaming.base.util import merge_index

s3 = boto3.client('s3')
def list_folders(bucket: str, prefix: str) -> List[str]:
    result = s3.list_objects_v2(Bucket=bucket, Prefix=prefix, Delimiter='/')
    return [o.get('Prefix') for o in result.get('CommonPrefixes', [])]

# # handle case of excessive nesting?
# def list_subfolders(bucket: str, prefix: str) -> List[str]:
#     result = s3.list_objects_v2(Bucket=bucket, Prefix=prefix, Delimiter='/')
#     subfolders = [o.get('Prefix') for o in result.get('CommonPrefixes', [])]
    
#     all_index_paths = []
#     for subfolder in subfolders:
#         sub_result = s3.list_objects_v2(Bucket=bucket, Prefix=subfolder, Delimiter='/')
#         sub_subfolders = [o.get('Prefix') for o in sub_result.get('CommonPrefixes', [])]
#         for sub_subfolder in sub_subfolders:
#             all_index_paths.append(f"{sub_subfolder}index.json")
    
#     return all_index_paths

def get_index_content(bucket: str, key: str) -> dict:
    try:
        response = s3.get_object(Bucket=bucket, Key=key)
        content = response['Body'].read().decode('utf-8')
        return json.loads(content)
    except ClientError as e:
        print(f"Error reading {key}: {e}")
        return None

def merge_s3_indices(bucket: str, prefix: str):
    folders = list_folders(bucket, prefix)
    index_urls = [f"s3://{bucket}/{folder}index.json" for folder in folders]

    merge_index(index_urls, f"s3://{bucket}/{prefix}", keep_local=False, download_timeout=60)


    print(f"Merged index uploaded to s3://{bucket}/{prefix}index.json") # it's called index.json by streaming


#%%
bucket = "counter-strike-data"
prefix = "vae_40M/"
#%%
merge_s3_indices(bucket, prefix)
# %%
