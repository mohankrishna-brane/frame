import os
import sys
import threading
import time

import cv2
import numpy as np
import requests

from .config.settings import FACE_MODEL, GO_ID


class ApiClient:
    def __init__(self, base_url):
        self.access_token = None
        self.session_data = {}
        self.baseurl = base_url
        self.session = requests.Session()

    def login(self, user_name, password):
        auth_payload = {"username": user_name, "password": password}
        try:
            response = self.session.post(
                f"{self.baseurl}/auth/login", auth_payload, timeout=5
            )
            response.raise_for_status()
            result = response.json()
            self.session_data["access_token"] = result["access_token"]
            self.session_data["tenant_id"] = result["user"]["tenant_id"]
            self.session_data["user_id"] = result["user"]["user_id"]
        except requests.RequestException as e:
            print(f"[ApiClient] Login failed: {e}")
            raise
        except Exception as e:
            print(f"[ApiClient] Unexpected error during login: {e}")
            raise

    def create_workflow_instance(self, go_id=GO_ID):
        req_payload = {
            "tenant_id": self.session_data.get("tenant_id"),
            "user_id": self.session_data.get("user_id"),
            "go_id": go_id,
            "test_mode": False,
        }
        req_url = f"{self.baseurl}/workflow_instances/?tenant_id={self.session_data.get('tenant_id')}"
        try:
            response = self.session.post(req_url, req_payload, timeout=5)
            response.raise_for_status()
            result = response.json()
            self.session_data["instance_id"] = result["instance_id"]
            self.session_data["instance_status"] = result.get("status", "Draft")
            return result["instance_id"]
        except requests.RequestException as e:
            print(f"[ApiClient] Create workflow instance failed: {e}")
            raise
        except Exception as e:
            print(
                f"[ApiClient] Unexpected error during workflow instance creation: {e}"
            )
            raise

    def start_workflow_instance(self):
        req_payload = {"user_id": self.session_data.get("user_id")}
        instance_id = self.session_data.get("instance_id")
        tenant_id = self.session_data.get("tenant_id")
        req_url = f"{self.baseurl}/workflow_instances/{instance_id}/start/?tenant_id={tenant_id}"
        try:
            response = self.session.post(req_url, req_payload, timeout=5)
            response.raise_for_status()
            result = response.json()
            self.session_data["instance_status"] = result.get("status", "Active")
            return True
        except requests.RequestException as e:
            print(f"[ApiClient] Start workflow instance failed: {e}")
            raise
        except Exception as e:
            print(
                f"[ApiClient] Unexpected error during starting workflow instance: {e}"
            )
            raise

    def execute_vector_search_workflow(self, go_id, embedding_vector):
        instance_id = self.session_data.get("instance_id")
        tenant_id = self.session_data.get("tenant_id")
        user_id = self.session_data.get("user_id")

        exec_url = f"{self.baseurl}/local_objectives/instances/{instance_id}/execute?tenant_id={tenant_id}"
        exec_payload = {
            "user_id": user_id,
            "tenant_id": tenant_id,
            "input_data": {"facialembeddings.embedding_vector": embedding_vector},
        }
        try:
            execution_result = self.session.post(exec_url, json=exec_payload, timeout=5)
            execution_result.raise_for_status()
            result = execution_result.json()
            print(f"[API] Execution Result: {result}")
        except requests.RequestException as e:
            print(f"[ApiClient] Execute vector search workflow failed: {e}")
            raise
        except Exception as e:
            print(
                f"[ApiClient] Unexpected error during vector search workflow execution: {e}"
            )
            raise
