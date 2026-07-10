
# Commandes NVIDIA connues (type 0x46 = 'F')
NVIDIA_IOCTLS = {
    0x27: "NV_ESC_RM_MAP_MEMORY",          # fd 12 = mapping mémoire GPU
    0x2a: "NV_ESC_RM_ALLOC_MEMORY",        # allocation mémoire
    0x2b: "NV_ESC_RM_FREE",                # libération ressource
    0x4e: "NV_ESC_RM_ALLOC_OBJECT",        # allocation objet CUDA
    0xc9: "NV_ESC_REGISTER_FD",            # enregistrement fd
    0xce: "NV_ESC_RM_CONTROL",             # contrôle général (GROS)
    0x49: "NV_ESC_RM_NVLOG",               # logging interne
    0x21: "NV_ESC_RM_GET_EVENT_DATA",      # récupération événements
    0x41: "NV_ESC_RM_VID_HEAP_CONTROL",    # gestion heap vidéo
    0x44: "NV_ESC_RM_I2C_ACCESS",          # accès I2C
    0x48: "NV_ESC_RM_IDLE_CHANNELS",       # idle channels
    0x17: "NV_ESC_RM_DUP_OBJECT",          # duplication objet
    0x1b: "NV_ESC_RM_SHARE",               # partage ressource
}

import re, sys

for line in sys.stdin:
    m = re.search(r'ioctl\((\d+), _IOC\([^,]+, (0x[0-9a-f]+), (0x[0-9a-f]+)', line)
    if m:
        fd, type_hex, cmd_hex = m.groups()
        cmd = int(cmd_hex, 16)
        name = NVIDIA_IOCTLS.get(cmd, f"UNKNOWN_0x{cmd:02x}")
        print(f"fd={fd:3s} | {name:40s} | {line.rstrip()}")
