#include "stdafx.h"

#include <assert.h> 
#include <chrono>
#include <iostream>

#include "BaseTrainer.h"
#include "3rdparty/json/json.hpp"

#include "FileIO/FileManager.h"

namespace Training
{
    using namespace FileIO;
    using namespace NeuralNetwork;
    using json = nlohmann::json;

    const std::string CONFIG_FILE_NAME = "configs.json";

    BaseTrainer::~BaseTrainer()
    {
        if (m_initialized)
        {
            m_nodeNetwork->destroyNetwork();
        }
    }

    bool BaseTrainer::setup()
    {
        FileManager::clearLogFile();

        if (!handleConfigSetup())
        {
            return false;
        }

        if (!setupTrainingData())
        {
            return false;
        }

        if (!setupNetwork())
        {
            return false;
        }

        if (!setupParameters())
        {
            return false;
        }

        m_initialized = true;
        return true;
    }

    bool BaseTrainer::handleConfigSetup()
    {
        if (!readConfigValues())
        {
            PRINT_ERROR("Failed to read config values");
        }

        if (!handleOptionValidation())
        {
            return false;
        }

        describeTrainer();
        return true;
    }

    bool BaseTrainer::readConfigValues()
    {
        json j;

        if (!FileManager::readJsonFromFile(CONFIG_FILE_NAME, j))
        {
            return false;
        }

        if (!j.is_object())
        {
            return false;
        }

        m_paramData.minRandomParamValue = j.at("min_random_parameter").get<double>();
        m_paramData.maxRandomParamValue = j.at("max_random_parameter").get<double>();

        m_paramData.mutationReplacementChance = j.at("mutation_replacement_chance").get<double>();
        m_paramData.mutationBonusChance = j.at("mutation_bonus_chance").get<double>();
        m_paramData.maxMutationReplacementChance = j.at("max_mutation_replacement_chance").get<double>();
        m_paramData.maxMutationBonusChance = j.at("max_mutation_bonus_chance").get<double>();
        m_paramData.mutationBonusScale = j.at("mutation_bonus_scale").get<double>();
        m_paramData.mutationRateIterationMultiplier = j.at("mutation_rate_iteration_multiplier").get<double>();

        m_paramData.numBestSetsKeptDuringEvolution = j.at("num_best_sets_kept_during_evolution").get<int>();
        m_paramData.numBestSetsMutatedDuringEvolution = j.at("num_best_sets_mutated_during_evolution").get<int>();
        m_paramData.numAddedRandomSetsDuringEvolution = j.at("num_random_sets_added_during_evolution").get<int>();

        m_numParamSets = j.at("num_param_sets").get<int>();
        m_numIterations = j.at("num_iterations").get<int>();
        m_numMatches = j.at("num_matches").get<int>();

        m_activationFunctionType = j.at("activation_function").get<std::string>();

        const auto& vec = j.at("num_hidden_nodes");
        for (json::const_iterator it = vec.begin(); it != vec.end(); ++it)
        {
            m_numHiddenNodes.push_back(*it);
        }

        return true;
    }

    bool BaseTrainer::handleOptionValidation() const
    {
        if (m_paramData.minRandomParamValue >= m_paramData.maxRandomParamValue)
        {
            std::ostringstream buffer;
            buffer << "Option mismatch: min. random parameter value must be smaller than max. random parameter value (currently "
                << m_paramData.minRandomParamValue << " and " << m_paramData.maxRandomParamValue << ", respectively)";
            PRINT_ERROR(buffer);
            return false;
        }

        if (m_numIterations > 1)
        {
            const int numSpecialEvolutionSets = m_paramData.numBestSetsKeptDuringEvolution + m_paramData.numBestSetsMutatedDuringEvolution + m_paramData.numAddedRandomSetsDuringEvolution;

            if (m_numParamSets <= numSpecialEvolutionSets)
            {
                std::ostringstream buffer;
                buffer << "Option mismatch: the number of sets added during the evolution step (kept, mutated and randomly added; currently "
                    << numSpecialEvolutionSets << ") may not be equal to or larger than the total number of sets ("
                    << m_numParamSets << "); otherwise iterating is pointless)";
                PRINT_ERROR(buffer);
                return false;
            }

            if (m_paramData.mutationBonusChance > 0 && m_paramData.mutationBonusScale == 0
                || m_paramData.mutationBonusChance <= 0 && m_paramData.mutationBonusScale != 0)
            {
                std::ostringstream buffer;
                buffer << "Warning: No mutation bonus will be applied during the evolution step because either the mutation chance ("
                    << m_paramData.mutationBonusChance << ") or the bonus scale (" << m_paramData.mutationBonusScale << ") is zero";
                std::cout << buffer.str() << std::endl;
                PRINT_LOG(buffer);
            }
        }

        return true;
    }

    void BaseTrainer::describeTrainer() const
    {
        std::ostringstream buffer;
        buffer << getName() << ": ";
        buffer << std::endl << "  #paramSets: " << m_numParamSets;
        buffer << std::endl << "  #matches: " << m_numMatches;
        buffer << std::endl << "  #iterations: " << m_numIterations;
        buffer << std::endl;
        PRINT_LOG(buffer);
    }

    bool BaseTrainer::setupTrainingData()
    {
        // nothing to do
        return true;
    }

    bool BaseTrainer::setupNetwork()
    {
        const NetworkSizeData sizeData = getNetworkSizeData();

        m_nodeNetwork = std::make_shared<NodeNetwork>();
        if (!m_nodeNetwork->createNetwork(sizeData, m_activationFunctionType))
        {
            PRINT_ERROR("Network creation failed!");
            return false;
        }

        return true;
    }

    NetworkSizeData BaseTrainer::getNetworkSizeData() const
    {
        NetworkSizeData sizeData;
        sizeData.numHiddenNodes = m_numHiddenNodes;
        return sizeData;
    }

    bool BaseTrainer::setupParameters()
    {
        // setup parameter manager
        m_paramData.numParams = m_nodeNetwork->getNumParameters();
        m_paramManager = std::make_shared<ParameterManager>(m_paramData);
        m_paramManager->describeParameterManager();

        // create N different parameter sets
        for (int k = 0; k < m_numParamSets; k++)
        {
            ParamSet pset;
            m_paramManager->fillWithRandomValues(pset.params);
            m_paramManager->addNewParamSet(pset);
        }

        return true;
    }

    void BaseTrainer::run()
    {
        const auto processStart = std::chrono::high_resolution_clock::now();

        if (!setup())
        {
            std::ostringstream buffer;
            buffer << "Failed to setup " << getName() << "!";
            PRINT_ERROR(buffer);
            return;
        }

        for (int i = 0; i < m_numIterations; i++)
        {
            handleTrainingIteration(i);
        }

        const auto processEnd = std::chrono::high_resolution_clock::now();
        const std::chrono::duration<double> elapsedSeconds = processEnd - processStart;

        std::ostringstream buffer;
        buffer << "Time taken: " << elapsedSeconds.count() << " seconds";
        std::cout << buffer.str();
        PRINT_LOG(buffer);

        m_paramManager->dumpDataToFile();
    }

    void BaseTrainer::handleTrainingIteration(int iteration)
    {
        std::ostringstream buffer;
        buffer << "Training iteration " << iteration;
        std::cout << buffer.str() << std::endl;
        PRINT_LOG(buffer);

        std::vector<int> currentIds;
        m_paramManager->getActiveParameterSetIds(currentIds);

        for (auto id : currentIds)
        {
            // reset network parameters
            buffer.clear();
            buffer.str("");
            buffer << "-----------------------------------------------"
                << std::endl << "Trying parameter set " << id << ": ";
            PRINT_LOG(buffer);

            ParamSet pset;
            m_paramManager->getParamSetForId(id, pset);

            if (pset.score != 0)
            {
                // skip parameter sets for which we already have a score from the previous run
                // but print the previous score again for convenience
                describeScoreForId(id);
                continue;
            }

            m_nodeNetwork->assignParameters(pset.params);
            handleNetworkComputation(id);
            describeScoreForId(id);

            // update score
            const double newScore = computeFinalScore(id);
            m_paramManager->setScore(id, newScore);
        }

        std::vector<int> bestSetIds;
        m_paramManager->getParameterSetIdsSortedByScore(bestSetIds);
        assert(!bestSetIds.empty());

        m_idsPerIteration.emplace(iteration, bestSetIds);

        ParamSet pset;
        m_paramManager->getParamSetForId(bestSetIds[0], pset);

        buffer.clear();
        buffer.str("");
        buffer << std::endl << "Best parameter set: " << bestSetIds[0] << ", with score: " << pset.score;
        PRINT_LOG(buffer);

        const bool requiresFurtherEvolution = (iteration < m_numIterations - 1);
        if (requiresFurtherEvolution)
        {
            handleParamSetEvolution();
        }
    }

    void BaseTrainer::handleNetworkComputation(int id)
    {
        // nothing to do
    }

    void BaseTrainer::describeScoreForId(int id) const
    {
        // nothing to do
    }

    double BaseTrainer::computeFinalScore(int id)
    {
        return 0;
    }

    void BaseTrainer::handleParamSetEvolution()
    {
        std::ostringstream buffer;

        std::vector<int> newParameterSetIds;
        if (!m_paramManager->evolveParameterSets(m_numParamSets, newParameterSetIds))
        {
            buffer.clear();
            buffer.str("");
            buffer << "Parameter evolution failed!";
            PRINT_ERROR(buffer);
            return;
        }

        std::vector<int> currentParameterSetIds;
        m_paramManager->getActiveParameterSetIds(currentParameterSetIds);

        for (auto id : currentParameterSetIds)
        {
            if (std::find(newParameterSetIds.begin(), newParameterSetIds.end(), id) == newParameterSetIds.end())
            {
                buffer.clear();
                buffer.str("");
                buffer << "Disabling parameter set " << id;
                PRINT_LOG(buffer);

                m_paramManager->setParameterSetActive(id, false);
            }
        }
    }
}
