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
        , m_nextId(0)
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
                    m_nextId = std::max<int>(id + 1, m_nextId);
                }
            }
        }

        return true;
    }

    bool ParameterManager::dumpDataToFile() const
    {
        json j;
        j["values"] = {};

        for (const auto& pset : m_paramSets)
        {
            json jps;
            jps["id"] = pset.first;
            jps["score"] = pset.second.score;
            jps["params"] = pset.second.params;

            j["values"].push_back(jps);
        }

        std::vector<int> sortedIds;
        getParameterSetIdsSortedByScore(sortedIds);
        j["bestIds"] = sortedIds;

        if (!sortedIds.empty())
        {
            ParamSet pset;
            getParamSetForId(sortedIds[0], pset);
            j["bestScore"] = pset.score;
        }

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
        m_paramSets.emplace(m_nextId++, pset);
    }

    void ParameterManager::setScore(int id, double score)
    {
        auto& found = m_paramSets.find(id);
        assert(found != m_paramSets.end());

        found->second.score = score;
    }

    void ParameterManager::setParameters(int id, const std::vector<double>& params)
    {
        auto& found = m_paramSets.find(id);
        assert(found != m_paramSets.end());

        found->second.params = params;
    }

    bool ParameterManager::getParamSetForId(int id, ParamSet& pset) const
    {
        const auto& found = m_paramSets.find(id);
        if (found == m_paramSets.end())
        {
            return false;
        }

        pset = found->second;
        return true;
    }

    void ParameterManager::getParameterSetIdsSortedByScore(std::vector<int>& idsSortedByScore) const
    {
        // create another map with score as the key, auto-sorted in descending order
        std::map<double, std::vector<int>, std::greater<double>> scoreMap;

        for (const auto& pset : m_paramSets)
        {
            const auto& foundScore = scoreMap.find(pset.second.score);
            if (foundScore == scoreMap.end())
            {
                scoreMap.emplace(pset.second.score, std::vector<int>({ pset.first }));
            }
            else
            {
                foundScore->second.push_back(pset.first);
            }
        }

        idsSortedByScore.clear();
        for (const auto& score : scoreMap)
        {
            for (auto id : score.second)
            {
                idsSortedByScore.push_back(id);
            }
        }
    }
}
