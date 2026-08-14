#include "RunLogger.h"

#include <iostream>
#include <stdexcept>

RunLogger::TeeBuf::TeeBuf(std::streambuf* primary, std::streambuf* secondary)
    : primaryBuf(primary), secondaryBuf(secondary) {}

int RunLogger::TeeBuf::overflow(int c) {
    if (c == EOF) return !EOF;
    const int p = primaryBuf ? primaryBuf->sputc(static_cast<char>(c)) : c;
    const int s = secondaryBuf ? secondaryBuf->sputc(static_cast<char>(c)) : c;
    if (c == '\n') {
        if (primaryBuf) primaryBuf->pubsync();
        if (secondaryBuf) secondaryBuf->pubsync();
    }
    return (p == EOF || s == EOF) ? EOF : c;
}

int RunLogger::TeeBuf::sync() {
    int p = primaryBuf ? primaryBuf->pubsync() : 0;
    int s = secondaryBuf ? secondaryBuf->pubsync() : 0;
    return (p == 0 && s == 0) ? 0 : -1;
}

RunLogger::RunLogger(const std::string& appName, const std::string& runName)
    : RunLogger(std::filesystem::path(PROJECT_ROOT) / "checkpoints" / appName, runName) {}

RunLogger::RunLogger(const std::filesystem::path& baseDir, const std::string& runName) {
    namespace fs = std::filesystem;
    runDirectory = baseDir / runName;
    fs::create_directories(runDirectory);

    logFile.open(runDirectory / "run.log");
    if (!logFile.is_open()) {
        throw std::runtime_error("Failed to open log file at " + (runDirectory / "run.log").string());
    }

    oldCoutBuf = std::cout.rdbuf();
    oldCerrBuf = std::cerr.rdbuf();
    coutTee = std::make_unique<TeeBuf>(oldCoutBuf, logFile.rdbuf());
    cerrTee = std::make_unique<TeeBuf>(oldCerrBuf, logFile.rdbuf());
    std::cout.rdbuf(coutTee.get());
    std::cerr.rdbuf(cerrTee.get());
}

RunLogger::~RunLogger() {
    std::cout.flush();
    std::cerr.flush();
    if (oldCoutBuf) std::cout.rdbuf(oldCoutBuf);
    if (oldCerrBuf) std::cerr.rdbuf(oldCerrBuf);
}

std::filesystem::path RunLogger::pathFor(const std::string& filename) const {
    return runDirectory / filename;
}
