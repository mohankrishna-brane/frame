import requests
import json

class APIClient:
    def __init__(self, base_url, tenant_id):
        self.base_url = base_url
        self.tenant_id = tenant_id
        self.user_id = None
        self.access_token = None
        self.session = requests.Session()

    def login(self, username, password):
        """
        Hits the login endpoint to retrieve a JWT/Access Token.
        Returns True if successful, False otherwise.
        """
        endpoint = f"{self.base_url}/auth/login"
        payload = {
            "username": username,
            "password": password
        }
        
        try:
            print(f"[API] Logging in as {username}...")
            response = self.session.post(endpoint, json=payload, timeout=5)
            response.raise_for_status() # Raise error for 4xx/5xx
            
            # Assuming standard JSON response: {"access_token": "xyz...", ...}
            data = response.json()
            self.access_token = data.get("access_token")
            self.tenant_id = data.get("user", {}).get("tenant_id", self.tenant_id)
            self.user_id = data.get("user", {}).get("user_id", self.tenant_id)

            print(f"[API] Tenant ID set to: {self.tenant_id}")
            print(f"[API] User ID set to: {self.user_id}")
            
            if self.access_token:
                # Update session headers so all future calls use this token
                self.session.headers.update({
                    "Authorization": f"Bearer {self.access_token}"
                })
                print("[API] Login Successful. Token acquired.")
                return True
            else:
                print("[API] Login Failed: No token in response.")
                return False
                
        except requests.exceptions.RequestException as e:
            print(f"[API] Login Connection Error: {e}")
            return False

    def execute_attendance_workflow(self, emp_id, go_id, embedding_vector):
        """
        Executes the 4-step sequence:
        1. Create Instance
        2. Start Instance
        3. Fetch Inputs
        4. Execute Workflow
        """
        if not self.access_token:
            print("[API] Workflow Aborted: No Token.")
            return

        print(f"[API] Starting Workflow for {emp_id}...")

        try:
            # --- STEP 1: CREATE INSTANCE ---
            create_url = f"{self.base_url}/workflow_instances/?tenant_id={self.tenant_id}"
            # TODO: Fill in your specific payload requirements here
            create_payload = {
                "go_id": go_id,
                "tenant_id": self.tenant_id,
                "user_id": self.user_id,
                "test_mode": "false"
                }
            
            resp = self.session.post(create_url, json=create_payload, timeout=5)
            resp.raise_for_status()
            data = resp.json()
            
            # Extract Instance ID (adjust key 'instance_id' if your API uses 'id' or 'uuid')
            instance_id = data.get("instance_id")
            if not instance_id:
                print("[API] Error: No instance_id returned in creation step.")
                return
            
            print(f"[API] Created Instance: {instance_id}")

            # --- STEP 2: START INSTANCE ---
            start_url = f"{self.base_url}/workflow_instances/{instance_id}/start/?tenant_id={self.tenant_id}"
            start_payload = {"user_id": self.user_id}
            self.session.post(start_url, json=start_payload, timeout=5).raise_for_status()
            
            # --- STEP 3: FETCH INPUTS ---
            inputs_url = f"{self.base_url}/local_objectives/instances/{instance_id}/inputs?tenant_id={self.tenant_id}"
            inputs_resp = self.session.get(inputs_url, timeout=5)
            inputs_resp.raise_for_status()
            # inputs_data = inputs_resp.json() # Use this if Step 4 needs data from Step 3

            # --- STEP 4: EXECUTE WORKFLOW ---
            exec_url = f"{self.base_url}/local_objectives/instances/{instance_id}/execute?tenant_id={self.tenant_id}"
            # TODO: Fill payload. This is likely where the heavy lifting happens.
            exec_payload = {
                "user_id": self.user_id,
                "tenant_id": self.tenant_id,
                "input_data": {
                    "facialembeddings.embedding_vector": embedding_vector
                }
            }
            
            execution_result = self.session.post(exec_url, json=exec_payload, timeout=5)
            # execution_result.raise_for_status()
            result = execution_result.json()
            print(f"[API] Execution Result: {result}")
            print(f"[API] Workflow COMPLETE for {emp_id} (Instance {instance_id})")

        except requests.exceptions.RequestException as e:
            print(f"[API] Workflow Failed at step: {e}")
            # Optional: Add error logging or response text printing for debugging
            # if e.response: print(e.response.text)