import sys
import os
from cryptography.fernet import Fernet

# --- KEY MANAGEMENT ---
# 1. Generate a secret key.
#    In a real application, you would save this key somewhere safe
#    and load it when you need it.
#    IMPORTANT: Keep this key secret!

#key = Fernet.generate_key()
#print(f"Generated Key: {key.decode()}")
#print("-" * 20)

key = "En2pJOBfM_e5G7iRuVmRYQqbYsDrj7HM-0t4si9f1qo="

# --- ENCRYPTION ---
# 2. Create a Fernet instance with your key. This is your "locked box".
cipher_suite = Fernet(key)

# 3. The message you want to protect. It must be in bytes.
if len(sys.argv) > 1:
    # The first actual argument is at index 1
    original_message = f"{sys.argv[1]}".encode('utf-8')
    print(f"Hello, {original_message}")
else:
    print("No original_message was provided.")
    os._exit(1)

# 4. Encrypt the message.
encrypted_message = cipher_suite.encrypt(original_message)

print(f"Original Message: {original_message.decode()}")
print(f"Encrypted Message: {encrypted_message}")
print("-" * 20)


# --- DECRYPTION ---
# 5. To decrypt, use the same Fernet instance (which holds the key).
decrypted_message = cipher_suite.decrypt(encrypted_message)

print(f"Decrypted Message: {decrypted_message.decode()}")
