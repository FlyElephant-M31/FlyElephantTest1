#ifndef PROFILER_HPP
#define PROFILER_HPP

#include <time.h>
#include <vector>
#include <iomanip>
#include <ostream>

class Profiler {
private:
    std::vector<double> _data;
    struct timespec _lastTime;
public:
    void tic();
    void toc(unsigned category);

    friend std::ostream &operator<<(std::ostream &out, const Profiler &profiler);
};

inline void Profiler::tic() {
    _lastTime = gettime();
}

inline void Profiler::toc(unsigned category) {
    static struct timespec current_ts;
    current_ts = gettime();
    if (category >= _data.size()) {
        _data.resize(category + 1);
    }
    _data[category] += (current_ts.tv_nsec - (double) _lastTime.tv_nsec) * 1.0e-9 + (current_ts.tv_sec - (double) _lastTime.tv_sec);
    _lastTime = current_ts;
}

inline std::ostream &operator<<(std::ostream &out, const Profiler &profiler) {
    out << "Profiler report:" << std::endl;
    for (int i = 0; i < profiler._data.size(); i++) {
        if (profiler._data[i] > 0.001) {
            out << "\tCategory " << std::setw(2) << std::setfill('0') << i << ": " <<
                    std::fixed << std::setprecision(3) << profiler._data[i] << " s" << std::endl;
        }
    }
    return out;
}

#endif