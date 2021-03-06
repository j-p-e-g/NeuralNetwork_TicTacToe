#include "stdafx.h"

#include <assert.h>
#include <iostream>

#include "FileIO/FileManager.h"
#include "TrainingMethodHandler.h"


namespace Training
{
    using namespace FileIO;
    using namespace Game;
    using namespace NeuralNetwork;

    TrainingMethodHandler::TrainingMethodHandler(std::shared_ptr<NodeNetwork>& network, std::shared_ptr<ParameterManager>& paramManager, std::shared_ptr<GameLogic>& gameLogic)
        : m_nodeNetwork(network)
        , m_paramManager(paramManager)
        , m_gameLogic(gameLogic)
    {
    }

    void TrainingMethodHandler::describeTrainingMethod() const
    {
        std::ostringstream buffer;
        buffer << "Training method: " << getName() << std::endl;
        PRINT_LOG(buffer);
    }

    void TrainingMethodHandler::iterationStart(int paramSetId)
    {
        m_currentParamSetId = paramSetId;
    }

    //---------------------------------------
    // ParameterEvolutionHandler
    //---------------------------------------
    ParameterEvolutionHandler::ParameterEvolutionHandler(std::shared_ptr<NeuralNetwork::NodeNetwork>& network, std::shared_ptr<ParameterManager>& paramManager, std::shared_ptr<Game::GameLogic>& gameLogic)
        : TrainingMethodHandler(network, paramManager, gameLogic)
    {
    }

    double ParameterEvolutionHandler::handleTrainingIteration(std::shared_ptr<BasePlayer>& player)
    {
        return 0;
    }

    void ParameterEvolutionHandler::iterationEnd(bool lastIteration)
    {
    }

    void ParameterEvolutionHandler::postIteration(bool lastIteration)
    {
        std::ostringstream buffer;

        std::vector<int> newParameterSetIds;
        if (!m_paramManager->evolveParameterSets(newParameterSetIds))
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

    //---------------------------------------
    // BackpropagationHandler
    //---------------------------------------
    BackpropagationHandler::BackpropagationHandler(std::shared_ptr<NeuralNetwork::NodeNetwork>& network, std::shared_ptr<ParameterManager>& paramManager, std::shared_ptr<Game::GameLogic>& gameLogic)
        : TrainingMethodHandler(network, paramManager, gameLogic)
    {
    }

    void BackpropagationHandler::describeTrainingMethod() const
    {
        std::ostringstream buffer;
        buffer << "Training method: " << getName();
        buffer << std::endl << "  Initial learning rate: " << m_learningRate;
        buffer << std::endl << "  Min. learning rate: " << m_minLearningRate;
        buffer << std::endl << "  Max. learning rate: " << m_maxLearningRate;
        buffer << std::endl << "  Learning rate factor (to correct for overshooting): " << m_learningRateFactor;
        buffer << std::endl;
        PRINT_LOG(buffer);
    }

    void BackpropagationHandler::iterationStart(int paramSetId)
    {
        TrainingMethodHandler::iterationStart(paramSetId);

        m_countTrainingSets = 0;
        m_errors.clear();
        m_parameterAdjustmentValues.clear();
        m_parameterAdjustmentValues.resize(m_nodeNetwork->getNumParameters());
    }

    double BackpropagationHandler::handleTrainingIteration(std::shared_ptr<Game::BasePlayer>& player)
    {
        m_countTrainingSets++;

        // let the game logic modify the output values to what it wants
        std::vector<double> outputValues;
        m_nodeNetwork->getOutputValues(outputValues, false);
        m_gameLogic->correctOutputValues(player->getPlayerId() == CS_PLAYER1 ? 0 : 1, outputValues);

        // then compute the error
        const double error = m_nodeNetwork->getTotalError(outputValues);
        m_errors.push_back(error);

        // and propagate back based on these values
        std::vector<double> tempAdjustmentValues;
        m_nodeNetwork->handleBackpropagation(outputValues, tempAdjustmentValues);
        assert(tempAdjustmentValues.size() == m_parameterAdjustmentValues.size());

        for (unsigned int k = 0; k < tempAdjustmentValues.size(); k++)
        {
            m_parameterAdjustmentValues[k] += tempAdjustmentValues[k];
        }

        return error;
    }

    void BackpropagationHandler::iterationEnd(bool lastIteration)
    {
        // calculate average error
        assert(!m_errors.empty());

        double avgError = 0;
        for (const auto& val : m_errors)
        {
            avgError += val;
        }

        avgError /= m_errors.size();

        m_paramManager->setError(m_currentParamSetId, avgError);

        std::ostringstream buffer;
        buffer << "  avg. error: " << avgError;

        if (lastIteration)
        {
            PRINT_LOG(buffer);
            return;
        }

        if (avgError > m_prevError)
        {
            // if the error gets worse, try a smaller learning rate
            m_learningRate *= m_learningRateFactor;
            if (m_learningRate < m_minLearningRate)
            {
                // but if the learning rate already is small and the error still gets worse,
                // reset the learning rate to a larger value and hope this will get us out
                // of a local minimum
                m_learningRate = m_maxLearningRate;
            }

            buffer << std::endl << "  new learning rate: " << m_learningRate;
        }

        m_prevError = avgError;

        // normalize adjustment values 
        // (large errors can result in huge adjustment values that cause the error to greatly increase)
        double maxAdjustmentValue = 0;
        double avgAdjustmentValue = 0;
        for (unsigned int k = 0; k < m_parameterAdjustmentValues.size(); k++)
        {
            m_parameterAdjustmentValues[k] /= m_countTrainingSets;
            maxAdjustmentValue = std::max(maxAdjustmentValue, std::abs(m_parameterAdjustmentValues[k]));
            avgAdjustmentValue += std::abs(m_parameterAdjustmentValues[k]);
        }

        avgAdjustmentValue /= m_parameterAdjustmentValues.size();

        buffer << std::endl << "  max. adjustment value: " << maxAdjustmentValue;
        buffer << std::endl << "  avg. adjustment value: " << avgAdjustmentValue << std::endl;
        PRINT_LOG(buffer);

        if (!m_normalizeAdjustmentValues || maxAdjustmentValue < 1)
        {
            // reset to 1, so it has no impact
            maxAdjustmentValue = 1;
        }

        // adjust parameters by average adjustment value
        std::vector<double> previousParams;
        m_nodeNetwork->getParameters(previousParams);

        ParamSet pset;
        for (unsigned int k = 0; k < m_parameterAdjustmentValues.size(); k++)
        {
            const double adjustmentValue = m_parameterAdjustmentValues[k] / maxAdjustmentValue;
            pset.params.push_back(previousParams[k] - m_learningRate * adjustmentValue);
        }

        int newId = m_paramManager->addNewParamSet(pset);
        m_paramManager->setParameterSetActive(m_currentParamSetId, false);
    }

    void BackpropagationHandler::postIteration(bool lastIteration)
    {
    }
}
