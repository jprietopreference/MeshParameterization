#pragma once
// Simple structured logger with file output + stdout
// Usage: LOG_INFO("message"); LOG_WARN("message"); LOG_ERROR("message");

#include <iostream>
#include <fstream>
#include <sstream>
#include <mutex>
#include <chrono>
#include <ctime>
#include <string>
#include <iomanip>

class MeshParamLogger {
public:
    enum Level { LVL_DEBUG, LVL_INFO, LVL_WARN, LVL_ERROR };

    static MeshParamLogger& instance() {
        static MeshParamLogger inst;
        return inst;
    }

    void init(const std::string& log_path = "", Level min_level = LVL_INFO) {
        std::lock_guard<std::mutex> lock(mtx_);
        min_level_ = min_level;
        if (!log_path.empty()) {
            file_.open(log_path, std::ios::app);
            if (file_.is_open()) {
                log_path_ = log_path;
                log(LVL_INFO, "Logger initialized: " + log_path);
            }
        }
    }

    void log(Level level, const std::string& msg) {
        if (level < min_level_) return;
        std::lock_guard<std::mutex> lock(mtx_);

        auto now = std::chrono::system_clock::now();
        auto t = std::chrono::system_clock::to_time_t(now);
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            now.time_since_epoch()) % 1000;

        std::tm tm_buf;
#ifdef _WIN32
        localtime_s(&tm_buf, &t);
#else
        localtime_r(&t, &tm_buf);
#endif

        std::ostringstream oss;
        oss << std::put_time(&tm_buf, "%Y-%m-%d %H:%M:%S")
            << '.' << std::setfill('0') << std::setw(3) << ms.count()
            << " [" << level_str(level) << "] " << msg;

        std::string line = oss.str();

        // stdout (always)
        (level >= LVL_WARN ? std::cerr : std::cout) << line << std::endl;

        // File (if open)
        if (file_.is_open()) {
            file_ << line << std::endl;
            // Rotate if > 10MB
            if (++line_count_ % 1000 == 0) {
                file_.seekp(0, std::ios::end);
                if (file_.tellp() > 10 * 1024 * 1024) {
                    file_.close();
                    std::string rotated = log_path_ + ".1";
                    std::remove(rotated.c_str());
                    std::rename(log_path_.c_str(), rotated.c_str());
                    file_.open(log_path_, std::ios::app);
                }
            }
        }
    }

private:
    Logger() = default;
    std::ofstream file_;
    std::string log_path_;
    std::mutex mtx_;
    Level min_level_ = LVL_INFO;
    int line_count_ = 0;

    static const char* level_str(Level l) {
        switch (l) {
            case LVL_DEBUG: return "DEBUG";
            case LVL_INFO:  return "INFO ";
            case LVL_WARN:  return "WARN ";
            case LVL_ERROR: return "ERROR";
        }
        return "?????";
    }
};

#define LOG_DEBUG(msg) MeshParamLogger::instance().log(MeshParamLogger::LVL_DEBUG, msg)
#define LOG_INFO(msg)  MeshParamLogger::instance().log(MeshParamLogger::LVL_INFO, msg)
#define LOG_WARN(msg)  MeshParamLogger::instance().log(MeshParamLogger::LVL_WARN, msg)
#define LOG_ERROR(msg) MeshParamLogger::instance().log(MeshParamLogger::LVL_ERROR, msg)
