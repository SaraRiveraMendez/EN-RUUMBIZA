#include <iostream>
#include <samplerate.h>

int main() {
    std::cout << "libsamplerate funcionando\n";

    std::cout << src_get_version() << std::endl;

    return 0;
}