#pragma once

#include <chrono>
#include <iostream>

class Perf {
    public:
    Perf(const std::string & name): mName(name), mStart(std::chrono::high_resolution_clock::now()) {}
    ~Perf() {
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> elapsed = end - mStart;
        std::cout << mName << " elapsed time: " << elapsed.count() << " ms" << std::endl;
    }
    private:
    const std::string mName;
    const std::chrono::time_point<std::chrono::high_resolution_clock> mStart;
};