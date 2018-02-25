#include "stdafx.h"

#include "ParameterManager.h"
#include "FileIO/FileManager.h"

#include "3rdparty/json/json.hpp"

#include <iostream>
#include <random>

namespace NeuralNetwork
{
    using namespace FileIO;
    using json = nlohmann::json;

    const std::string DATA_FILE_NAME = "data/values.json";

    ParameterManager::ParameterManager(const ParameterManagerData pmData)
        : m_paramData(pmData)
        , m_highestId(0)
    {
    }

    bool ParameterManager::readDataFromFile()
    {
        json j;

        if (!FileManager::readJsonFromFile(DATA_FILE_NAME, j))
        {
            return false;
        }

        if (!j.is_object())
        {
            return false;
        }

        // debug output
        std::string dump = j.dump();
        std::cout << "json content: " << dump << std::endl;

        if (j.find("values") != j.end())
        {
            auto& values = j.at("values");
            for (json::iterator it = values.begin(); it != values.end(); ++it)
            {
                std::cout << *it << '\n';
                if (it->find("id") != it->end())
                {
                    const int id = it->at("id");
                    m_highestId = std::max<int>(id, m_highestId);
                }
            }
        }

        return true;
    }

    bool ParameterManager::dumpDataToFile() const
    {
        json j;
        j["score"] = -0.279;
        j["id"] = "test";
        j["params"] = { 9.32, -4.205, 0.55 };

        const std::string fileName = "data/test.json";
        return FileManager::writeJsonToFile(fileName, j);
    }

    void ParameterManager::fillWithRandomValues(std::vector<double>& params) const
    {
        params.clear();
        params.reserve(m_paramData.numParams);

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<double> rndDist(m_paramData.minValue, std::nextafter(m_paramData.maxValue, DBL_MAX));

        for (int k = 0; k < m_paramData.numParams; k++)
        {
            params.push_back(rndDist(gen));
        }
    }

    void ParameterManager::addNewParamSet(const ParamSet& pset)
    {
        m_paramSets.emplace(++m_highestId, pset);
    }
}
