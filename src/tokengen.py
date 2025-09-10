import base64
import json
import os
import time
import urllib.parse
from azure.identity import DefaultAzureCredential
from azure.mgmt.postgresqlflexibleservers import PostgreSQLManagementClient

from dotenv import load_dotenv
load_dotenv()

TOKEN_FILE = "aad_token.txt"

class TokenValidation:
    def __init__(self):  # Fixed: was missing underscores
        self.aad_in_use = os.environ.get("AZURE_USE_AAD")
        self.dbhost = self.get_environ_variable("PGHOST")
        self.dbuser = urllib.parse.quote(self.get_environ_variable("PGUSER"))
        self.token = None  # Store the token here

        if self.aad_in_use == "True":
            self.subscription_id = self.get_environ_variable("AZURE_SUBSCRIPTION_ID")
            self.resource_group_name = self.get_environ_variable("AZURE_RESOURCE_GROUP")
            self.server_name = (
                self.dbhost.split(".", 1)[0] if "." in self.dbhost else self.dbhost
            )
            self.credential = DefaultAzureCredential()
            self.postgresql_client = PostgreSQLManagementClient(
                self.credential, self.subscription_id
            )
        # Don't call get_password() in init to avoid duplicate token checks
        self.password = None

    @staticmethod
    def get_environ_variable(name: str):
        """Helper function to get environment variable or raise an error."""
        value = os.environ.get(name)
        if value is None:
            raise EnvironmentError(f"Environment variable {name} not found.")
        return value

    def get_password(self) -> str:
        """Get password based on the auth mode set, caching the token in a file if AAD is used."""
        if self.aad_in_use == "True":
            token = self.load_token_from_file()
            if not token or not self.is_token_valid(token):
                token = self.credential.get_token(
                    "https://ossrdbms-aad.database.windows.net/.default"
                ).token
                self.save_token_to_file(token)
            self.token = token
            return token
        else:
            return self.get_environ_variable("PGPASSWORD")

    def load_token_from_file(self) -> str:
        try:
            with open(TOKEN_FILE, "r") as f:
                return f.read().strip()
        except Exception:
            return None

    def save_token_to_file(self, token: str):
        with open(TOKEN_FILE, "w") as f:
            f.write(token)

    def is_token_valid(self, token: str) -> bool:
        """Check if the given JWT token is still valid."""
        try:
            payload_part = token.split('.')[1]
            padding = '=' * (-len(payload_part) % 4)
            payload_part += padding
            payload = json.loads(base64.urlsafe_b64decode(payload_part))
            exp = payload.get('exp')
            if exp is None:
                return False
            current_time = int(time.time())
            return current_time < exp
        except Exception:
            return False

    def check_and_refresh_token(self):
        """Check if token is expired and refresh if needed."""
        if self.aad_in_use == "True":
            token = self.load_token_from_file()
            
            if not token:
                print("No token found, generating new token...")
                # Force refresh by getting a new token
                new_token = self.credential.get_token(
                    "https://ossrdbms-aad.database.windows.net/.default"
                ).token
                self.save_token_to_file(new_token)
                print("New token generated and saved.")
                return True
            
            if not self.is_token_valid(token):
                print("Token is expired, generating new token...")
                # Force refresh by getting a new token
                new_token = self.credential.get_token(
                    "https://ossrdbms-aad.database.windows.net/.default"
                ).token
                self.save_token_to_file(new_token)
                print("New token generated and saved.")
                return True
            else:
                # Calculate time until expiration
                try:
                    payload_part = token.split('.')[1]
                    padding = '=' * (-len(payload_part) % 4)
                    payload_part += padding
                    payload = json.loads(base64.urlsafe_b64decode(payload_part))
                    exp = payload.get('exp')
                    current_time = int(time.time())
                    seconds_until_expiry = exp - current_time
                    minutes_until_expiry = seconds_until_expiry // 60
                    remaining_seconds = seconds_until_expiry % 60
                    
                    exp_time_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(exp))
                    print(f"Token is valid. Expires at: {exp_time_str} (in {minutes_until_expiry} minutes and {remaining_seconds} seconds)")
                    return False
                except Exception as e:
                    print(f"Error checking token validity: {e}")
                    return False
        else:
            print("AAD is not in use, no token refresh needed.")
            return False

    def check_token_validity(self):
        """Checks if the current token is valid and prints the result and expiration time."""
        if self.aad_in_use == "True":
            token = self.get_password()  # This will use the cached token if valid
            # JWT tokens are in the format header.payload.signature
            try:
                payload_part = token.split('.')[1]
                # Pad base64 string if necessary
                padding = '=' * (-len(payload_part) % 4)
                payload_part += padding
                payload = json.loads(base64.urlsafe_b64decode(payload_part))
                exp = payload.get('exp')
                if exp is None:
                    print("Token does not have an expiration (exp) field.")
                    return
                current_time = int(time.time())
                exp_time_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(exp))
                
                seconds_until_expiry = exp - current_time
                if current_time < exp:
                    minutes_until_expiry = seconds_until_expiry // 60
                    remaining_seconds = seconds_until_expiry % 60
                    print("Token is valid.")
                    print(f"Token expires at: {exp_time_str} (in {minutes_until_expiry} minutes and {remaining_seconds} seconds)")
                else:
                    minutes_since_expiry = abs(seconds_until_expiry) // 60
                    remaining_seconds = abs(seconds_until_expiry) % 60
                    print("Token has expired.")
                    print(f"Token expired at: {exp_time_str} ({minutes_since_expiry} minutes and {remaining_seconds} seconds ago)")
            except Exception as e:
                print(f"Failed to decode or validate token: {e}")
        else:
            print("AAD is not in use, no token to validate.")

if __name__ == "__main__":
    obj = TokenValidation()
    obj.check_and_refresh_token()
