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

    const std::string DATA_FILE_NAME = "params.json";

    ParameterManager::ParameterManager(const ParameterManagerData pmData)
        : m_paramData(pmData)
        , m_nextId(0)
    {
        assert(m_paramData.minRandomParamValue < m_paramData.maxRandomParamValue);

        // skip this step if there's no chance of actually tweaking any parameters
        m_executeMutationStep = (m_paramData.mutationReplacementChance > 0) || (m_paramData.mutationBonusChance > 0 && m_paramData.mutationBonusScale != 0);

        // initialize mutation rates
        updateEffectiveMutationRates(-1);
    }

    void ParameterManager::describeParameterManager() const
    {
        std::ostringstream buffer;
        buffer << "ParameterManager: ";
        buffer << std::endl << "  #parameters: " << m_paramData.numParams;
        buffer << std::endl << "  random param values picked within [" << m_paramData.minRandomParamValue << ", " << m_paramData.maxRandomParamValue << "]";
        buffer << std::endl << "  mutation replacement chance during evolution: " << m_paramData.mutationReplacementChance;
        buffer << std::endl << "  mutation bonus chance during evolution: " << m_paramData.mutationBonusChance;
        buffer << std::endl << "  mutation bonus scale during evolution: " << m_paramData.mutationBonusScale;
        buffer << std::endl << "  mutation bonus within [" << m_paramData.minRandomParamValue * m_paramData.mutationBonusScale 
                            << ", " << m_paramData.maxRandomParamValue * m_paramData.mutationBonusScale << "]";
        buffer << std::endl << "  mutation rate multiplier for iterations without improvement: " << m_paramData.mutationRateIterationMultiplier;
        buffer << std::endl << "  max. mutation replacement chance during evolution: " << m_paramData.maxMutationReplacementChance;
        buffer << std::endl << "  max. mutation bonus chance during evolution: " << m_paramData.maxMutationBonusChance;
        buffer << std::endl << "  number of best sets kept during evolution: " << m_paramData.numBestSetsKeptDuringEvolution;
        buffer << std::endl << "  number of best sets mutated during evolution: " << m_paramData.numBestSetsMutatedDuringEvolution;
        buffer << std::endl << "  number of random sets added during evolution: " << m_paramData.numAddedRandomSetsDuringEvolution;
        buffer << std::endl;
        PRINT_LOG(buffer);
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

        return FileManager::writeJsonToFile(DATA_FILE_NAME, j);
    }

    void ParameterManager::fillWithRandomValues(std::vector<double>& params) const
    {
        params.clear();
        params.reserve(m_paramData.numParams);

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<double> rndDist(m_paramData.minRandomParamValue, std::nextafter(m_paramData.maxRandomParamValue, DBL_MAX));

        for (int k = 0; k < m_paramData.numParams; k++)
        {
            params.push_back(rndDist(gen));
        }
    }

    int ParameterManager::addNewParamSet(const ParamSet& pset)
    {
        m_paramSets.emplace(m_nextId, pset);
        return m_nextId++;
    }

    void ParameterManager::removeParameterSetForId(int id)
    {
        m_paramSets.erase(id);
    }

    void ParameterManager::setScore(int id, double score)
    {
        auto& found = m_paramSets.find(id);
        assert(found != m_paramSets.end());

        found->second.score = score;
    }

    void ParameterManager::setParameterSetActive(int id, bool active)
    {
        auto& found = m_paramSets.find(id);
        assert(found != m_paramSets.end());

        found->second.active = active;
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

    void ParameterManager::getActiveParameterSetIds(std::vector<int>& ids) const
    {
        ids.clear();

        for (const auto& pset : m_paramSets)
        {
            if (pset.second.active)
            {
                ids.push_back(pset.first);
            }
        }
    }

    void ParameterManager::getParameterSetIdsSortedByScore(std::vector<int>& idsSortedByScore) const
    {
        // create another map with score as the key, auto-sorted in descending order
        std::map<double, std::vector<int>, std::greater<double>> scoreMap;

        for (const auto& pset : m_paramSets)
        {
            if (!pset.second.active)
            {
                continue;
            }

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

    bool ParameterManager::evolveParameterSets(int totalNumberOfSets, std::vector<int>& newParameterSetIds)
    {
        std::ostringstream buffer;
        buffer << std::endl << "Evolving parameter sets...";

        std::vector<int> bestParameterSetIds;
        getParameterSetIdsSortedByScore(bestParameterSetIds);
        assert(bestParameterSetIds.size() > 1);

        updateEffectiveMutationRates(bestParameterSetIds[0]);

        if (m_numIterationsBestIdUnchanged > 0)
        {
            buffer << std::endl << " num. consecutive iterations without improvement: " << m_numIterationsBestIdUnchanged;
            buffer << std::endl << " - effective mutation replacement chance: " << m_effectiveMutationReplacementChance;
            buffer << std::endl << " - effective mutation bonus chance: " << m_effectiveMutationBonusChance;
        }

        std::map<double, int> probabilityMap;
        fillParameterSetProbabilityMap(probabilityMap);
        assert(probabilityMap.size() > 1);

        assert(m_paramData.numBestSetsKeptDuringEvolution + m_paramData.numBestSetsKeptDuringEvolution + m_paramData.numAddedRandomSetsDuringEvolution < totalNumberOfSets);

        // keep best parameter sets
        for (int k = 0; k < m_paramData.numBestSetsKeptDuringEvolution; k++)
        {
            buffer << std::endl << "Keeping best set " << bestParameterSetIds[k];
            newParameterSetIds.push_back(bestParameterSetIds[k]);
        }

        if (m_executeMutationStep)
        {
            // create heavily mutated version of the best parameter sets
            for (int k = 0; k < m_paramData.numBestSetsMutatedDuringEvolution; k++)
            {
                const int bestId = bestParameterSetIds[k];

                ParamSet pset;
                if (m_executeMutationStep && !createMutatedParameterSet(bestId, pset))
                {
                    return false;
                }

                // in the unlikely case that this new set is identical to the previous one, don't bother adding it
                ParamSet oldSet;
                if (getParamSetForId(bestId, oldSet))
                {
                    if (oldSet.params == pset.params)
                    {
                        continue;
                    }
                }

                const int newSetId = addNewParamSet(pset);
                buffer << std::endl << "Mutating best set " << bestId << " resulted in new param set " << newSetId;
                newParameterSetIds.push_back(newSetId);
            }
        }

        // add new random sets
        for (int k = 0; k < m_paramData.numAddedRandomSetsDuringEvolution; k++)
        {
            ParamSet pset;
            fillWithRandomValues(pset.params);

            const int newRandomSetId = addNewParamSet(pset);
            buffer << std::endl << "Adding new random set " << newRandomSetId;
            newParameterSetIds.push_back(newRandomSetId);
        }

        PRINT_LOG(buffer);

        // fill the remaining slots by combining previous sets
        while (newParameterSetIds.size() < totalNumberOfSets)
        {
            const int id1 = getIdByProbability(probabilityMap);
            const int id2 = getIdByProbability(probabilityMap);
            assert(id1 >= 0);
            assert(id2 >= 0);

            if (id1 == id2)
            {
                continue;
            }

            ParamSet pset;
            if (!createCrossoverParameterSet(id1, id2, pset))
            {
                return false;
            }

            const int newSetId = addNewParamSet(pset);

            buffer.clear();
            buffer.str("");
            buffer << "Crossover between " << id1 << " and " << id2 << " resulted in new param set " << newSetId;
            PRINT_LOG(buffer);

            newParameterSetIds.push_back(newSetId);
        }

        return true;
    }

    void ParameterManager::updateEffectiveMutationRates(int newBestSetId)
    {
        m_effectiveMutationReplacementChance = m_paramData.mutationReplacementChance;
        m_effectiveMutationBonusChance = m_paramData.mutationBonusChance;

        if (newBestSetId == -1 || m_paramData.mutationRateIterationMultiplier <= 0)
        {
            return;
        }

        if (newBestSetId == m_currentBestSetId)
        {
            m_numIterationsBestIdUnchanged++;
        }
        else
        {
            m_numIterationsBestIdUnchanged = 0;
        }

        m_currentBestSetId = newBestSetId;

        if (m_numIterationsBestIdUnchanged <= 0)
        {
            return;
        }

        m_effectiveMutationReplacementChance *= (1 + m_numIterationsBestIdUnchanged * m_paramData.mutationRateIterationMultiplier);
        if (m_effectiveMutationReplacementChance > m_paramData.maxMutationReplacementChance)
        {
            m_effectiveMutationReplacementChance = m_paramData.maxMutationReplacementChance;
        }

        m_effectiveMutationBonusChance *= (1 + m_numIterationsBestIdUnchanged * m_paramData.mutationRateIterationMultiplier);
        if (m_effectiveMutationBonusChance > m_paramData.maxMutationBonusChance)
        {
            m_effectiveMutationBonusChance = m_paramData.maxMutationBonusChance;
        }
    }

    void ParameterManager::fillParameterSetProbabilityMap(std::map<double, int> &probabilityMap)
    {
        std::vector<int> bestIds;
        getParameterSetIdsSortedByScore(bestIds);
        assert(!bestIds.empty());

        ParamSet worstScore;
        getParamSetForId(bestIds[bestIds.size() - 1], worstScore);
        const double lowestScore = worstScore.score;

        std::map<int, double> modifiedScores;

        double sum = 0;
        for (auto id : bestIds)
        {
            ParamSet pset;
            getParamSetForId(id, pset);

            const double tempScore = pset.score - lowestScore;
            if (tempScore > 0)
            {
                sum += tempScore;
                modifiedScores.emplace(id, tempScore);
            }
        }

        assert(sum > 0);

        double summedChance = 0;
        for (auto ms : modifiedScores)
        {
            summedChance += ms.second / sum;
            probabilityMap.emplace(summedChance, ms.first);
        }
    }

    int ParameterManager::getIdByProbability(const std::map<double, int>& probabilityMap)
    {
        assert(!probabilityMap.empty());

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<double> rndChance(0, 1);

        const double chance = rndChance(gen);
        for (const auto& prob : probabilityMap)
        {
            if (chance <= prob.first)
            {
                return prob.second;
            }
        }

        return -1;
    }

    bool ParameterManager::createMutatedParameterSet(int id, ParamSet& pset) const
    {
        ParamSet pidset;

        if (!getParamSetForId(id, pidset))
        {
            std::ostringstream buffer;
            buffer << "Unable to create mutated parameter set for invalid id " << id;
            PRINT_ERROR(buffer);
            return false;
        }

        pset.params.clear();
        for (const auto& param : pidset.params)
        {
            pset.params.push_back(getMutatedValue(param));
        }

        return  true;
    }

    bool ParameterManager::createCrossoverParameterSet(int id1, int id2, ParamSet& pset) const
    {
        ParamSet p1;
        ParamSet p2;

        if (!getParamSetForId(id1, p1) || !getParamSetForId(id2, p2))
        {
            std::ostringstream buffer;
            buffer << "Unable to create crossover set: one or more of the parent ids (" << id1 << ", " << id2 << ") is invalid";
            PRINT_ERROR(buffer);
            return false;
        }

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<double> rndChance(0, 1);

        pset.params.clear();
        for (int k = 0; k < m_paramData.numParams; k++)
        {
            // use value from either of the parents
            const double param = (rndChance(gen) < 0.5 ? p1.params[k] : p2.params[k]);
            pset.params.push_back(getMutatedValue(param));
        }

        return  true;
    }

    double ParameterManager::getMutatedValue(double param) const
    {
        if (!m_executeMutationStep)
        {
            return param;
        }

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<double> rndChance(0, 1);
        std::uniform_real_distribution<double> rndDist(m_paramData.minRandomParamValue, std::nextafter(m_paramData.maxRandomParamValue, DBL_MAX));

        if (m_effectiveMutationReplacementChance > 0 && rndChance(gen) <= m_effectiveMutationReplacementChance)
        {
            return rndDist(gen);
        }

        if (m_effectiveMutationBonusChance > 0 && rndChance(gen) <= m_effectiveMutationBonusChance)
        {
            // randomly tweak the parameter, but ensure that it's still between [min, max]
            const double newParam = param + rndDist(gen) * m_paramData.mutationBonusScale;
            return std::min(m_paramData.maxRandomParamValue, std::max(m_paramData.minRandomParamValue, newParam));
        }

        return param;
    }
}
