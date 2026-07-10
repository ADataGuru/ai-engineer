#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdio.h>
#include <sys/ioctl.h>

int ioctl(int fd, unsigned long request, ...) {
    static int (*real_ioctl)(int, unsigned long, ...) = NULL;
    if (!real_ioctl) real_ioctl = dlsym(RTLD_NEXT, "ioctl");

    // Type 0x46 = NVIDIA
    if (((request >> 8) & 0xff) == 0x46) {
        unsigned char cmd = request & 0xff;
        fprintf(stderr, "[HOOK] NVIDIA ioctl fd=%d cmd=0x%02x\n", fd, cmd);
    }

    va_list args;
    va_start(args, request);
    void* arg = va_arg(args, void*);
    va_end(args);
    return real_ioctl(fd, request, arg);
}

gcc -shared -fPIC -o hook_cuda.so hook_