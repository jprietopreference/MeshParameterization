#pragma once
// Cross-platform helpers for subprocess management and path resolution

#include <string>
#include <vector>
#include <functional>
#include <cstdlib>

#ifdef _WIN32
#  define WIN32_LEAN_AND_MEAN
#  include <windows.h>
#else
#  include <unistd.h>
#  include <sys/types.h>
#  include <sys/wait.h>
#  include <signal.h>
#  include <spawn.h>
#  include <poll.h>
#  include <limits.h>
#endif

// Platform-specific executable suffix
#ifdef _WIN32
static const char* EXE_SUFFIX = ".exe";
#else
static const char* EXE_SUFFIX = "";
#endif

// Get directory of the current executable
inline std::string get_exe_dir() {
    char buf[4096] = {};
#ifdef _WIN32
    GetModuleFileNameA(nullptr, buf, sizeof(buf));
#elif defined(__linux__)
    ssize_t len = readlink("/proc/self/exe", buf, sizeof(buf) - 1);
    if (len > 0) buf[len] = '\0';
#elif defined(__APPLE__)
    uint32_t size = sizeof(buf);
    _NSGetExecutablePath(buf, &size);
#endif
    std::string path(buf);
    auto pos = path.find_last_of("/\\");
    return (pos != std::string::npos) ? path.substr(0, pos + 1) : "./";
}

// Find bench CLI executable next to server binary
inline std::string find_bench_exe() {
    std::string dir = get_exe_dir();
    std::string name = std::string("meshparam_bench") + EXE_SUFFIX;
    std::string full = dir + name;
    // Check if exists
    FILE* f = fopen(full.c_str(), "r");
    if (f) { fclose(f); return full; }
    // Fallback: try PATH
    return name;
}

// Subprocess handle (cross-platform)
struct SubprocessHandle {
    std::string method;
#ifdef _WIN32
    HANDLE hProcess = INVALID_HANDLE_VALUE;
    HANDLE hThread = INVALID_HANDLE_VALUE;
#else
    pid_t pid = -1;
#endif
    std::string tmp_json;
    bool valid = false;
};

// Spawn a subprocess: returns handle
inline SubprocessHandle spawn_subprocess(const std::string& cmd) {
    SubprocessHandle h;
#ifdef _WIN32
    STARTUPINFOA si = {}; si.cb = sizeof(si);
    si.dwFlags = STARTF_USESHOWWINDOW; si.wShowWindow = SW_HIDE;
    PROCESS_INFORMATION pi = {};
    if (CreateProcessA(nullptr, const_cast<char*>(cmd.c_str()),
                      nullptr, nullptr, FALSE, CREATE_NO_WINDOW,
                      nullptr, nullptr, &si, &pi)) {
        h.hProcess = pi.hProcess;
        h.hThread = pi.hThread;
        h.valid = true;
    }
#else
    pid_t pid = fork();
    if (pid == 0) {
        // Child: exec via shell
        execl("/bin/sh", "sh", "-c", cmd.c_str(), (char*)nullptr);
        _exit(127);
    } else if (pid > 0) {
        h.pid = pid;
        h.valid = true;
    }
#endif
    return h;
}

// Wait for subprocess with timeout (milliseconds). Returns exit code, -1 on timeout.
inline int wait_subprocess(SubprocessHandle& h, int timeout_ms) {
#ifdef _WIN32
    DWORD rc = WaitForSingleObject(h.hProcess, timeout_ms);
    if (rc == WAIT_TIMEOUT) return -1;
    DWORD exit_code = 1;
    GetExitCodeProcess(h.hProcess, &exit_code);
    return (int)exit_code;
#else
    // Poll with timeout
    int elapsed = 0;
    int step = 50; // ms
    while (elapsed < timeout_ms) {
        int status;
        pid_t ret = waitpid(h.pid, &status, WNOHANG);
        if (ret > 0) {
            return WIFEXITED(status) ? WEXITSTATUS(status) : 1;
        }
        usleep(step * 1000);
        elapsed += step;
    }
    return -1; // timeout
#endif
}

// Kill subprocess
inline void kill_subprocess(SubprocessHandle& h) {
#ifdef _WIN32
    TerminateProcess(h.hProcess, 1);
    WaitForSingleObject(h.hProcess, 5000);
#else
    if (h.pid > 0) {
        kill(h.pid, SIGKILL);
        waitpid(h.pid, nullptr, 0);
    }
#endif
}

// Close subprocess handles
inline void close_subprocess(SubprocessHandle& h) {
#ifdef _WIN32
    if (h.hProcess != INVALID_HANDLE_VALUE) CloseHandle(h.hProcess);
    if (h.hThread != INVALID_HANDLE_VALUE) CloseHandle(h.hThread);
    h.hProcess = INVALID_HANDLE_VALUE;
    h.hThread = INVALID_HANDLE_VALUE;
#else
    h.pid = -1;
#endif
    h.valid = false;
}

// Run subprocess synchronously with timeout. Returns exit code.
inline int run_subprocess(const std::string& cmd, int timeout_ms = 30000) {
    SubprocessHandle h = spawn_subprocess(cmd);
    if (!h.valid) return -1;
    int rc = wait_subprocess(h, timeout_ms);
    if (rc == -1) {
        kill_subprocess(h);
        rc = -1;
    }
    close_subprocess(h);
    return rc;
}

// Install signal handler for graceful shutdown
inline void install_shutdown_handler(std::function<void()> on_shutdown) {
#ifdef _WIN32
    static std::function<void()> s_handler = on_shutdown;
    SetConsoleCtrlHandler([](DWORD type) -> BOOL {
        if (type == CTRL_C_EVENT || type == CTRL_CLOSE_EVENT || type == CTRL_SHUTDOWN_EVENT) {
            if (s_handler) s_handler();
            return TRUE;
        }
        return FALSE;
    }, TRUE);
#else
    static std::function<void()> s_handler = on_shutdown;
    struct sigaction sa = {};
    sa.sa_handler = [](int) { if (s_handler) s_handler(); };
    sigaction(SIGINT, &sa, nullptr);
    sigaction(SIGTERM, &sa, nullptr);
#endif
}
