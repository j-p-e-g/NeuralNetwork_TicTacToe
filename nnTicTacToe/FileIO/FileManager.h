#pragma once

#include <fstream>
#include <string>

#include "../3rdparty/json/json.hpp"

namespace FileIO
{
    class FileManager
    {
    public:
        static bool fileExists(const std::string& fileName);
        static bool getRelativeFilePath(const std::string& filePath, std::string& relativePath);
        static bool openInFileStream(const std::string& fileName, std::ifstream& ifs);
        static bool openOutFileStream(const std::string& fileName, std::ofstream& ofs);

        static bool readJsonFromFile(const std::string& fileName, nlohmann::json& jsonObject);
        static bool writeJsonToFile(const std::string& fileName, const nlohmann::json& jsonObject);
    };
}
