#define _CRT_SECURE_NO_WARNINGS

#include <sycl/sycl.hpp>
#include <array>
#include <fstream>
#include <chrono>

using namespace std;
using namespace chrono;

constexpr array<uint8_t, 11> Qn = {6, 5, 4, 3, 2, 7, 6, 5, 4, 3, 2};

template<size_t Size>
constexpr array<int64_t, Size> initiate_div() {
    array<int64_t, Size> ret{};
    int64_t a = 1;
    for (int i = 0; i < Size; ++i) {
        ret[Size - 1 - i] = a;
        a *= 10;
    }
    return ret;
}

constexpr auto div_arr = initiate_div<11>();
constexpr auto snprintf_arr = initiate_div<12>();


int main(int argc, char *argv[]) {
    if (argc != 2) {
        cerr << "please specify filename!!" << endl;
        cerr << argv[0] << " " << "output_filename.txt" << endl;
        exit(-1);
    }
    FILE *out = fopen(argv[1], "wb");
    if (!out) {
        cerr << "error opening file:" << argv[1] << endl;
        exit(-2);
    }
    auto syclDevice = sycl::device(sycl::default_selector_v);
    static sycl::queue syclQueue(syclDevice);
    cerr << "SYCL Device Name: " << syclDevice.get_info<sycl::info::device::name>() << endl;
    cerr << "SYCL Device Vendor: " << syclDevice.get_info<sycl::info::device::vendor>() << endl;
    const auto begin = system_clock::now();
    const int64_t chunk = 1e8L;
    const int64_t all = 1e11L;
    setvbuf(out, nullptr, _IOFBF, chunk * 13);
    auto output_string = sycl::malloc_device<char>(chunk * 13 + 1, syclQueue);
    auto malloc_device_local = static_cast<char *>(malloc((chunk * 13 + 1) * sizeof(char)));
    for (int64_t i = 0; i < all; i += chunk) {
        syclQueue.parallel_for(chunk, [=](auto j) {
            int64_t input = i + j;
            int16_t acc = 0;
            for (int k = 0; k < 11; ++k) {
                acc += Qn[k] * ((input / div_arr[k]) % 10); // NOLINT(*-narrowing-conversions)
            }
            acc %= 11;
            if (acc <= 1) {
                acc = 0;
            } else {
                acc = 11 - acc; // NOLINT(*-narrowing-conversions)
            }
            auto comp_val = input * 10 + acc;
            for (int k = 0; k < 12; ++k) {
                output_string[13 * j + k] = '0' + (comp_val / snprintf_arr[k]) % 10;
            }
            output_string[13 * j + 12] = '\n';

        });
        syclQueue.single_task([=]() { output_string[chunk * 13] = '\0'; });
        syclQueue.wait();
        syclQueue.memcpy(malloc_device_local, output_string, sizeof(char) * (chunk * 13 + 1)).wait();
        fwrite(malloc_device_local, sizeof(char), chunk * 13, out);
        fprintf(stderr, "\r%.2Lf%%\t%.3LfGbps",
                static_cast<long double>(i + chunk) / (all * 1e-2),
                static_cast<long double>((i + chunk) * 13 * 8) / 1000'000'000.0L / (static_cast<long double>(duration_cast<milliseconds>((system_clock::now() - begin)).count()) / 1000));
        if (i > chunk * 100) break;
    }
    sycl::free(output_string, syclQueue);
    free(malloc_device_local);
    fflush(out);
    fprintf(stderr, "\n%lld sec elapsed.\n", duration_cast<seconds>((system_clock::now() - begin)).count());
    fclose(out);
}