#include <sycl/sycl.hpp>
#include <array>
#include <fstream>

using namespace std;

constexpr array<uint8_t, 11> Qn = {6, 5, 4, 3, 2, 7, 6, 5, 4, 3, 2};


template<size_t Size>
constexpr array<int64_t, Size> initiate_div() {
    array<int64_t, Size> ret;
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
    std::ofstream out(argv[1], std::ios::out | std::ios::trunc);
    auto syclDevice = sycl::device(sycl::default_selector_v);
    static sycl::queue syclQueue(syclDevice);
    cerr << "SYCL Device Name: " << syclDevice.get_info<sycl::info::device::name>() << endl;
    cerr << "SYCL Device Vendor: " << syclDevice.get_info<sycl::info::device::vendor>() << endl;
    const int64_t chunk = 1e5;
    for (int64_t i = 0; i < 1e12L; i += chunk) {
        auto output_string = sycl::malloc_shared<char>(chunk * 13 + 1, syclQueue);
        syclQueue.parallel_for(chunk, [=](auto j) {
            int64_t input = i + j;
            int16_t acc = 0;
            for (int k = 0; k < 11; ++k) {
                acc += Qn[k] * ((input / div_arr[k]) % 10);
            }
            acc %= 11;
            if (acc <= 1) {
                acc = 0;
            } else {
                acc = 11 - acc;
            }
            auto comp_val = input * 10 + acc;
            for (int k = 0; k < 12; ++k) {
                output_string[13 * j + k] = '0' + (comp_val / snprintf_arr[k]) % 10;
            }
            output_string[13 * j + 12] = '\n';

        }).wait();
        syclQueue.single_task([=]() { output_string[chunk * 13] = '\0'; }).wait();
        out.write(output_string, chunk * 13);
        out.flush();
        sycl::free(output_string, syclQueue);
//        if (i > chunk * 2) break;
    }
}