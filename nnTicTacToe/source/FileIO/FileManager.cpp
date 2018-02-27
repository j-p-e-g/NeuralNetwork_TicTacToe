#include "stdafx.h"

#include "FileManager.h"

#include <iostream>
#include <sys/stat.h>

namespace FileIO
{
    using json = nlohmann::json;

    const int MAX_NESTED_PATH_LOOKUP_ATTEMPTS = 3;
    const std::string DATA_PATH = "data/";
    const std::string LOGFILE_NAME = "logfile.txt";
    
    void FileManager::clearLogFile()
    {
        std::string relativePath;
        if (!getRelativeDataFilePath(LOGFILE_NAME, relativePath))
        {
            return;
        }

        // don't write anything, opening and closing will be enough to clear it
        std::ofstream ofs;
        openOutFileStream(relativePath, ofs);

        if (!ofs.is_open())
        {
            std::cerr << "Unable to open log file '" << relativePath << "'" << std::endl;
        }
    }

    void FileManager::addToLogFile(const std::ostringstream& stream)
    {
        addToLogFile(stream.str());
    }

    void FileManager::addToLogFile(const std::string& msg)
    {
        std::string relativePath;
        if (!getRelativeDataFilePath(LOGFILE_NAME, relativePath))
        {
            return;
        }

        std::ofstream ofs;
        openOutFileStream(relativePath, ofs, std::ofstream::out | std::ofstream::app);
        if (!ofs.is_open())
        {
            //std::cerr << "Unable to open log file '" << relativePath << "'" << std::endl;
            return;
        }

        ofs << msg << std::endl;
    }

    void FileManager::printErrorMessage(const std::ostringstream& stream)
    {
        printErrorMessage(stream.str());
    }

    void FileManager::printErrorMessage(const std::string& msg)
    {
        const std::string errorMsg = "ERROR: " + msg;
        std::cerr << errorMsg << std::endl;
        addToLogFile(errorMsg);
    }

    bool FileManager::openInFileStream(const std::string& filePath, std::ifstream& ifs)
    {
        ifs = std::ifstream(filePath);
        if (ifs.is_open())
        {
            return true;
        }

        return false;
    }

    bool FileManager::openOutFileStream(const std::string& filePath, std::ofstream& ofs, int mode)
    {
        ofs = std::ofstream(filePath, mode);
        if (ofs.is_open())
        {
            return true;
        }

        return false;
    }

    bool FileManager::getRelativeFilePath(const std::string& filePath, std::string& relativePath)
    {
        relativePath = filePath;

        for (int k = 0; k < MAX_NESTED_PATH_LOOKUP_ATTEMPTS; k++)
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

    bool FileManager::getRelativeDataFilePath(const std::string& fileName, std::string& relativePath)
    {
        if (!getRelativeFilePath(DATA_PATH, relativePath))
        {
            std::cerr << "Unable to find path '" << DATA_PATH << "'" << std::endl;
            return false;
        }

        relativePath += fileName;
        return true;
    }

    bool FileManager::readJsonFromFile(const std::string& fileName, nlohmann::json& jsonObject)
    {
        std::string relativePath;
        if (!getRelativeDataFilePath(fileName, relativePath))
        {
            return false;
        }

        std::ifstream ifs;
        if (!openInFileStream(relativePath, ifs))
        {
            std::ostringstream buffer;
            buffer << "Failed to open file '" << relativePath.c_str() << "' for reading";
            PRINT_ERROR(buffer);
            return false;
        }

        jsonObject = json::parse(ifs);
        return true;
    }

    bool FileManager::writeJsonToFile(const std::string& fileName, const nlohmann::json& jsonObject)
    {
        std::string relativePath;
        if (!getRelativeDataFilePath(fileName, relativePath))
        {
            return false;
        }

        std::ofstream ofs;
        if (openOutFileStream(relativePath, ofs))
        {
            ofs << std::setw(4) << jsonObject << std::endl;
            return true;
        }

        std::ostringstream buffer;
        buffer << "Failed to open file '" << relativePath.c_str() << "' for writing";
        PRINT_ERROR(buffer);
        return false;
    }
}
