import keyring

# This saves the key securely in your OS Credential Manager
keyring.set_password("SivanAI", "HF_TOKEN", "your_huggingface_token_here")
print("Key securely stored in OS Keychain!")