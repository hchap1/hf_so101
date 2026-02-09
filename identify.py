from serial.tools import list_ports

def find_so101_port():
    for port in list_ports.comports():
        if port.vid in (0x1A86, 0x10C4, 0x0403):
            if "USB" in (port.description or ""):
                return port.device

    return None
