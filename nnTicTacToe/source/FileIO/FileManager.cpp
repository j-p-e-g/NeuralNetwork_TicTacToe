#include "stdafx.h"

#include "FileManager.h"

#include <iostream>
#include <sys/stat.h>

namespace FileIO
{
    using json = nlohmann::json;

    const int MAX_FIND_FILE_ATTEMPTS = 3;

    bool FileManager::openInFileStream(const std::string& fileName, std::ifstream& ifs)
    {
        std::string relativePath;
        if (getRelativeFilePath(fileName, relativePath))
        {
            ifs = std::ifstream(relativePath);
            if (ifs.is_open())
            {
                return true;
            }
        }

        return false;
    }

    bool FileManager::openOutFileStream(const std::string& fileName, std::ofstream& ofs)
    {
        std::string relativePath;
        if (getRelativeFilePath(fileName, relativePath))
        {
            ofs = std::ofstream(relativePath);
            if (ofs.is_open())
            {
                return true;
            }
        }

        return false;
    }

    bool FileManager::getRelativeFilePath(const std::string& filePath, std::string& relativePath)
    {
        relativePath = filePath;

        for (int k = 0; k < MAX_FIND_FILE_ATTEMPTS; k++)
        {
            if (fileExists(relativePath))
            {
                return true;
            }

            relativePath = "../" + relativePath;
        }

        return false;
    }

    bool FileManager::fileExists(const std::string& filename)
    {
        // implementation copied from StackOverflow
        struct stat buf;
        if (stat(filename.c_str(), &buf) != -1)
        {
            return true;
        }

        return false;
    }

    bool FileManager::readJsonFromFile(const std::string& fileName, nlohmann::json& jsonObject)
    {
        std::ifstream ifs;
        if (!openInFileStream(fileName, ifs))
        {
            std::cerr << "Failed to open file '" << fileName << "' for reading" << std::endl;
            return false;
        }

        jsonObject = json::parse(ifs);
        return true;
    }

    bool FileManager::writeJsonToFile(const std::string& fileName, const nlohmann::json& jsonObject)
    {
        std::ofstream ofs;
        if (openOutFileStream(fileName, ofs))
        {
            ofs << std::setw(4) << jsonObject << std::endl;
            return true;
        }

        std::cerr << "Failed to open file '" << fileName << "' for writing" << std::endl;
        return false;
    }
}
