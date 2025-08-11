from cryptography.fernet import Fernet

def get_msg():
    cipher_suite = Fernet("En2pJOBfM_e5G7iRuVmRYQqbYsDrj7HM-0t4si9f1qo=")
    msg = b'gAAAAABokrejhZN6shG57d_DVL3_dSbmxsctH1WNl2rOUsAbq6zmKoiGJq0Pzevuh96lWFMIIJ5VCVlrfLjfyT5esnCUPSaToQJ-HdlDLm8qPoUwD55O5Xh1A7SEWuvKq10H3GQDlKsb'
    msg = cipher_suite.decrypt(msg).decode()
    return msg
