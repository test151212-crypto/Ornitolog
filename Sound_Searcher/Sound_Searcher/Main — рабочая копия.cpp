#include <mpg123.h>
#include <fftw3.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <iomanip>
#include <string>
#include <fstream>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ==========================================
// НАСТРОЙКИ
// ==========================================
const std::string INPUT_PATH_FILE = "D:\\input_way.txt";        // Файл, откуда читаем путь к MP3
const std::string OUTPUT_FILE = "D:\\frequencies.txt"; // Куда сохранять результат

// --- Analysis Parameters ---
const size_t FFT_SIZE = 4096;
const size_t HOP_SIZE = FFT_SIZE / 2;
const double PERSISTENCE_RATIO = 0.7;
const double ACTIVE_THRESHOLD_DB = -40.0;

// Вспомогательная функция для удаления кавычек из пути
std::string cleanPath(std::string path) {
    if (!path.empty() && path.front() == '"' && path.back() == '"') {
        return path.substr(1, path.size() - 2);
    }
    return path;
}

int main() {
    std::string input_file;

    // ===== 0. Чтение пути из input.txt =====
    std::ifstream inPathFile(INPUT_PATH_FILE);
    if (!inPathFile.is_open()) {
        std::cerr << "Error: Could not open " << INPUT_PATH_FILE << "\n";
        std::cerr << "Please create a file named '" << INPUT_PATH_FILE << "' and put the MP3 path inside it.\n";
        std::cout << "\nPress Enter to exit...";
        std::cin.get();
        return 1;
    }

    std::getline(inPathFile, input_file);
    inPathFile.close();

    // Убираем возможные кавычки и пробелы в начале/конце
    input_file = cleanPath(input_file);

    if (input_file.empty()) {
        std::cerr << "Error: " << INPUT_PATH_FILE << " is empty.\n";
        std::cout << "\nPress Enter to exit...";
        std::cin.get();
        return 1;
    }

    std::cout << "Reading path from " << INPUT_PATH_FILE << "\n";
    std::cout << "Analyzing file: " << input_file << "\n";
    std::cout << "Results will be saved to: " << OUTPUT_FILE << "\n\n";

    // ===== 1. Initialize mpg123 =====
    mpg123_init();
    int err;
    mpg123_handle* mh = mpg123_new(nullptr, &err);
    if (!mh) {
        std::cerr << "Error: mpg123_new failed\n";
        mpg123_exit();
        return 1;
    }

    long rate = 0;
    int channels = 0, encoding = 0;

    // ===== 2. First open: get native rate =====
    if (mpg123_open(mh, input_file.c_str()) != MPG123_OK) {
        std::cerr << "Failed to open file: " << input_file << "\n";
        mpg123_delete(mh);
        mpg123_exit();
        return 1;
    }
    if (mpg123_getformat(mh, &rate, &channels, &encoding) != MPG123_OK) {
        std::cerr << "Failed to get format\n";
        mpg123_close(mh);
        mpg123_delete(mh);
        mpg123_exit();
        return 1;
    }
    mpg123_close(mh);

    // ===== 3. Force mono float32 at native rate =====
    mpg123_format_none(mh);
    if (mpg123_format(mh, rate, MPG123_MONO, MPG123_ENC_FLOAT_32) != MPG123_OK) {
        std::cerr << "Failed to set format\n";
        mpg123_delete(mh);
        mpg123_exit();
        return 1;
    }

    // ===== 4. Reopen with new format =====
    if (mpg123_open(mh, input_file.c_str()) != MPG123_OK) {
        std::cerr << "Failed to reopen file\n";
        mpg123_delete(mh);
        mpg123_exit();
        return 1;
    }
    if (mpg123_getformat(mh, &rate, &channels, &encoding) != MPG123_OK ||
        encoding != MPG123_ENC_FLOAT_32 || channels != 1) {
        std::cerr << "Failed to get expected format (mono float32)\n";
        mpg123_close(mh);
        mpg123_delete(mh);
        mpg123_exit();
        return 1;
    }

    // ===== 5. Decode entire file into samples =====
    std::vector<float> samples;
    std::vector<unsigned char> buffer(8192);
    size_t done;

    while (true) {
        int ret = mpg123_read(mh, buffer.data(), buffer.size(), &done);
        if (done > 0) {
            const float* fbuf = reinterpret_cast<const float*>(buffer.data());
            size_t count = done / sizeof(float);
            samples.insert(samples.end(), fbuf, fbuf + count);
        }
        if (ret == MPG123_DONE) break;
        if (ret != MPG123_OK) {
            std::cerr << "Warning: read error (" << ret << ")\n";
            break;
        }
    }

    mpg123_close(mh);
    mpg123_delete(mh);
    mpg123_exit();

    if (samples.size() < FFT_SIZE) {
        std::cerr << "File is too short for analysis\n";
        std::cout << "\nPress Enter to exit...";
        std::cin.get();
        return 1;
    }

    // ===== 6. Prepare FFTW =====
    size_t num_blocks = (samples.size() - FFT_SIZE) / HOP_SIZE + 1;
    size_t num_bins = FFT_SIZE / 2 + 1;

    std::vector<double> input(FFT_SIZE);
    std::vector<fftw_complex> output(num_bins);
    std::vector<double> energy_accum(num_bins, 0.0);
    std::vector<size_t> active_count(num_bins, 0);

    fftw_plan plan = fftw_plan_dft_r2c_1d(FFT_SIZE, input.data(),
        output.data(), FFTW_ESTIMATE);
    if (!plan) {
        std::cerr << "Failed to create FFT plan\n";
        return 1;
    }

    // Hann window
    std::vector<double> window(FFT_SIZE);
    for (size_t i = 0; i < FFT_SIZE; ++i)
        window[i] = 0.5 * (1.0 - cos(2.0 * M_PI * i / (FFT_SIZE - 1)));

    double threshold_ratio = pow(10.0, ACTIVE_THRESHOLD_DB / 10.0);

    // ===== 7. Block analysis =====
    for (size_t b = 0; b < num_blocks; ++b) {
        size_t start = b * HOP_SIZE;
        for (size_t i = 0; i < FFT_SIZE; ++i)
            input[i] = samples[start + i] * window[i];

        fftw_execute(plan);

        // Block peak (excluding DC)
        double block_max = 0.0;
        for (size_t k = 1; k < num_bins; ++k) {
            double re = output[k][0], im = output[k][1];
            double mag2 = re * re + im * im;
            if (mag2 > block_max) block_max = mag2;
        }
        if (block_max <= 0.0) continue; // silence

        double block_threshold = block_max * threshold_ratio;

        for (size_t k = 1; k < num_bins; ++k) {
            double re = output[k][0], im = output[k][1];
            double mag2 = re * re + im * im;
            energy_accum[k] += mag2;
            if (mag2 > block_threshold) active_count[k]++;
        }
    }

    fftw_destroy_plan(plan);

    // ===== 8. Extract stable frequencies (INTEGER Hz) =====
    double freq_per_bin = static_cast<double>(rate) / FFT_SIZE;
    std::vector<std::pair<int, double> > stable_freqs;

    for (size_t k = 1; k < num_bins; ++k) {
        double persistence = static_cast<double>(active_count[k]) / num_blocks;
        if (persistence >= PERSISTENCE_RATIO) {
            double avg_energy = energy_accum[k] / num_blocks;
            int freq_hz = static_cast<int>(std::round(k * freq_per_bin));
            stable_freqs.push_back(std::make_pair(freq_hz, avg_energy));
        }
    }

    // Sort descending by energy
    std::sort(stable_freqs.begin(), stable_freqs.end(),
        [](const std::pair<int, double>& a, const std::pair<int, double>& b) {
            return a.second > b.second;
        });

    // ===== 9. Output to Console and File =====
    std::ofstream outFile(OUTPUT_FILE);
    if (!outFile.is_open()) {
        std::cerr << "Error: Could not create file at " << OUTPUT_FILE << "\n";
    }
    else {
        outFile << "Stable frequencies (active in >= "
            << PERSISTENCE_RATIO * 100 << "% of blocks):\n";
    }

    std::cout << "Stable frequencies (active in >= "
        << PERSISTENCE_RATIO * 100 << "% of blocks):\n";

    outFile << std::fixed << std::setprecision(2);
    std::cout << std::fixed << std::setprecision(2);

    if (stable_freqs.empty()) {
        std::cout << "  Not found\n";
        if (outFile.is_open()) outFile << "  Not found\n";
    }
    else {
        double max_energy = stable_freqs.front().second;
        for (size_t i = 0; i < stable_freqs.size(); ++i) {
            int freq = stable_freqs[i].first;
            double energy = stable_freqs[i].second;
            double rel_db = 10.0 * log10(energy / max_energy + 1e-30);

            std::cout << "  " << freq << " Hz  (" << rel_db
                << " dB relative to the strongest)\n";

            if (outFile.is_open()) {
                outFile << "  " << freq << " Hz  (" << rel_db
                    << " dB relative to the strongest)\n";
            }
        }
    }

    if (outFile.is_open()) {
        outFile.close();
        std::cout << "\nSuccessfully saved to " << OUTPUT_FILE << "\n";
    }

    std::cout << "\nPress Enter to exit...";
    std::cin.ignore();
    std::cin.get();
    return 0;
}