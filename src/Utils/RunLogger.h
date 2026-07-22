#pragma once

#include <filesystem>
#include <fstream>
#include <memory>
#include <streambuf>
#include <string>

class RunLogger {
public:
    RunLogger(const std::string& appName, const std::string& runName);
    RunLogger(const std::filesystem::path& baseDir, const std::string& runName);
    ~RunLogger();

    RunLogger(const RunLogger&) = delete;
    RunLogger& operator=(const RunLogger&) = delete;

    const std::filesystem::path& runDir() const { return runDirectory; }
    std::filesystem::path pathFor(const std::string& filename) const;
    std::ostream& fileStream() { return logFile; }

private:
    class TeeBuf : public std::streambuf {
    public:
        TeeBuf(std::streambuf* primary, std::streambuf* secondary);
    protected:
        int overflow(int c) override;
        int sync() override;
    private:
        std::streambuf* primaryBuf;
        std::streambuf* secondaryBuf;
    };

    std::filesystem::path runDirectory;
    std::ofstream logFile;
    std::streambuf* oldCoutBuf = nullptr;
    std::streambuf* oldCerrBuf = nullptr;
    std::unique_ptr<TeeBuf> coutTee;
    std::unique_ptr<TeeBuf> cerrTee;
};
