    
import pandas as pd
import os
import sys
from datetime import datetime
from typing import Tuple
import time
import random
import string

from google.cloud import aiplatform, language
from google.cloud import bigquery
from google.cloud import storage
from google_cloud_pipeline_components.preview import model_evaluation
from kfp import compiler

import vertexai
from vertexai.preview.language_models import TextGenerationModel

def prompt_func(prompt_input: str):
    """Prompts designed to enrich Product description information"""
    prompt = f"""
        You are a leading digital marketer working for a top retail organization. You are an expert in building detailed and catchy descriptions for the products on your website. 

        Context: {prompt_input}

        Generate ONLY the product description in English that highlights the product's features using the above "Context" information. 
        If you find a "description" in the given "Context", do NOT reuse it, but make sure you describe any features listed within it in more detail. 
        Do NOT repeat sentences. The generated description should strictly be about the provided product. 
        Correct product type, number of items contained in the the product as well as product features such as color should be followed. 
        Any product features that are not present in the input should not be present in the generated description.
        Hyperbolic text, over promising or guarantees are to be avoided.
        The generated description should be at least 50 words long, preferably at least 150. 
        The generated description MUST NOT use special characters or any Markdown or JSON syntax. 

        New Detailed Product Description:"""
    return prompt



# Generate a uuid of a specifed length(default=8)
def generate_uuid(length: int = 8) -> str:
    """Generates a custom-length UUID-like string.

    Uses a combination of lowercase letters and digits to create a 
    randomized string resembling a shortened UUID (Universally Unique Identifier).

    Args:
        length (int, optional): The desired length of the generated string. 
                                Defaults to 8 characters.

    Returns:
        str: The generated UUID-like string.
    """
    return "".join(random.choices(string.ascii_lowercase + string.digits, k=length))

UUID = generate_uuid()


# Gemma deployment

def get_job_name_with_datetime(prefix: str) -> str:
    """Gets the job name with date time when triggering deployment jobs."""
    return prefix + datetime.now().strftime("_%Y%m%d_%H%M%S")


def deploy_model_vllm(
    model_name: str,
    model_id: str,
    service_account: str,
    machine_type: str = "g2-standard-12",
    accelerator_type: str = "NVIDIA_L4",
    accelerator_count: int = 1,
    max_model_len: int = 8192,
    dtype: str = "bfloat16",
) -> Tuple[aiplatform.Model, aiplatform.Endpoint]:
    """Deploys models with vLLM on GPU in Vertex AI."""
    endpoint = aiplatform.Endpoint.create(display_name=f"{model_name}-endpoint")

    vllm_args = [
        "--host=0.0.0.0",
        "--port=7080",
        f"--model={model_id}",
        f"--tensor-parallel-size={accelerator_count}",
        "--swap-space=16",
        "--gpu-memory-utilization=0.9",
        f"--max-model-len={max_model_len}",
        f"--dtype={dtype}",
        "--disable-log-stats",
    ]

    env_vars = {
        "MODEL_ID": model_id,
    }
    # if HF_TOKEN:
    #     env_vars["HF_TOKEN"] = HF_TOKEN

    model = aiplatform.Model.upload(
        display_name=model_name,
        serving_container_image_uri=VLLM_DOCKER_URI,
        serving_container_command=["python", "-m", "vllm.entrypoints.api_server"],
        serving_container_args=vllm_args,
        serving_container_ports=[7080],
        serving_container_predict_route="/generate",
        serving_container_health_route="/ping",
        serving_container_environment_variables=env_vars,
        serving_container_shared_memory_size_mb=(16 * 1024),  # 16 GB
        serving_container_deployment_timeout=7200,
    )

    model.deploy(
        endpoint=endpoint,
        machine_type=machine_type,
        accelerator_type=accelerator_type,
        accelerator_count=accelerator_count,
        deploy_request_timeout=1800,
        service_account=service_account,
        sync=True,
        enable_access_logging=True,
    )
    return model, endpoint


def save_csv_gcs(BUCKET_NAME: str, evaluation_dataset_name: str):
    """
    Saves a CSV file to a Google Cloud Storage bucket.

    Args:
        BUCKET_NAME (str):  The name of the GCS bucket (excluding the 'gs://' prefix).
        evaluation_dataset_name (str):  The filename (without extension) to use for the saved CSV.

    """
    
    # save to GCS
    storage_client = storage.Client()
    bucket = storage_client.bucket(BUCKET_NAME[5:])
    blob = bucket.blob(f"data/{evaluation_dataset_name}.csv")
    blob.upload_from_filename(f"{evaluation_dataset_name}.csv")

    print(f"File uploaded to cloud storage in {BUCKET_NAME}/data/{evaluation_dataset_name}.csv")
    
    
    
def save_jsonl_gcs(BUCKET_NAME: str, evaluation_dataset_name: str):
    """
    Saves a JSON Lines (.jsonl) file to a Google Cloud Storage bucket.

    Args:
        BUCKET_NAME (str):  The name of the GCS bucket (excluding the 'gs://' prefix).
        evaluation_dataset_name (str):  The filename (without extension) to use for the saved JSONL file.
    """
    
    # save to GCS 
    storage_client = storage.Client()
    bucket = storage_client.bucket(BUCKET_NAME[5:])
    blob = bucket.blob(f"data/{evaluation_dataset_name}.jsonl")
    blob.upload_from_filename(f"{evaluation_dataset_name}.jsonl")

    print(f"File uploaded to cloud storage in {BUCKET_NAME}/data/{evaluation_dataset_name}.jsonl")
