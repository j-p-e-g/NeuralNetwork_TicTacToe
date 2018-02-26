#pragma once

#include <fstream>
#include <string>

#include "../3rdparty/json/json.hpp"

namespace FileIO
{
#define PRINT_LOG(msg)    FileManager::addToLogFile(msg)
#define PRINT_ERROR(msg)   FileManager::printErrorMessage(msg)

    class FileManager
    {
    public:
        static void clearLogFile();
        static void addToLogFile(const std::string& msg);
        static void addToLogFile(const std::ostringstream& stream);
        static void printErrorMessage(const std::string& msg);
        static void printErrorMessage(const std::ostringstream& stream);

        static bool readJsonFromFile(const std::string& fileName, nlohmann::json& jsonObject);
        static bool writeJsonToFile(const std::string& fileName, const nlohmann::json& jsonObject);

    private:
        static bool fileExists(const std::string& fileName);
        static bool getRelativeDataFilePath(const std::string& fileName, std::string& relativePath);
        static bool getRelativeFilePath(const std::string& filePath, std::string& relativePath);
        static bool openInFileStream(const std::string& filePath, std::ifstream& ifs);
        static bool openOutFileStream(const std::string& filePath, std::ofstream& ofs, int mode = std::ofstream::out);
    };
}
