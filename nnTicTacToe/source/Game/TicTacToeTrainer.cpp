#include "stdafx.h"

#include <assert.h> 
#include <iostream>

#include "TicTacToeTrainer.h"

#include "GameLogic.h"
#include "NeuralNetwork/NodeNetwork.h"
#include "NeuralNetwork/ParameterManager.h"

namespace Game
{
    using namespace NeuralNetwork;

    TicTacToeTrainer::TicTacToeTrainer()
        : m_minParamValue(-10.f)
        , m_maxParamValue(10.f)
        , m_numParamSets(5)
        , m_numMatches(10)
    {
    }

    TicTacToeTrainer::~TicTacToeTrainer()
    {
        if (m_initialized)
        {
            m_nodeNetwork->destroyNetwork();
        }
    }

    bool TicTacToeTrainer::setup()
    {
        m_gameLogic = std::make_shared<TicTacToeLogic>();

        NetworkSizeData sizeData;
        m_gameLogic->getRequiredNetworkSize(sizeData);

        m_nodeNetwork = std::make_shared<NodeNetwork>();
        if (!m_nodeNetwork->createNetwork(sizeData))
        {
            std::cerr << "Network creation failed!" << std::endl;
            return false;
        }

        ParameterManagerData pmData;
        pmData.numParams = m_nodeNetwork->getNumParameters();
        pmData.minValue = m_minParamValue;
        pmData.maxValue = m_maxParamValue;

        m_paramManager = std::make_shared<ParameterManager>(pmData);

        // create N different parameter sets
        for (int k = 0; k < m_numParamSets; k++)
        {
            ParamSet pset;
            m_paramManager->fillWithRandomValues(pset.params);
            m_paramManager->addNewParamSet(pset);
        }

        m_initialized = true;
        return true;
    }

    void TicTacToeTrainer::run()
    {
        if (!setup())
        {
            std::cerr << "Failed to setup TicTacToeTrainer!" << std::endl;
            return;
        }

        for (int k = 0; k < m_numParamSets; k++)
        {
            // reset network parameters
            std::cout << std::endl << "Trying parameter set " << k << ": " << std::endl;

            ParamSet pset;
            m_paramManager->getParamSetForId(k, pset);
            m_nodeNetwork->assignParameters(pset.params);

            const double score = playMatch();
            std::cout << "Score: " << score << std::endl;
            m_paramManager->setScore(k, score);
        }

        std::vector<int> bestSetIds;
        m_paramManager->getParameterSetIdsSortedByScore(bestSetIds);
        assert(!bestSetIds.empty());

        ParamSet pset;
        m_paramManager->getParamSetForId(bestSetIds[0], pset);

        std::cout << "Best parameter set: " << bestSetIds[0] << " (score: " << pset.score << ")" << std::endl;
    }

    double TicTacToeTrainer::playMatch()
    {
        // reset board
        m_gameLogic->initBoard();

        int turnCount = 1;
        GameState state = GS_ONGOING;

        do
        {
            std::cout << "Turn " << turnCount << ": " << std::endl;
            state = playOneTurn();
            turnCount++;
        }
        while (state == GS_ONGOING);

        return computeMatchScore(turnCount, state);
    }

    double TicTacToeTrainer::computeMatchScore(int numTurns, GameState finalGameState)
    {
        double score = 0.0;
        score += numTurns;

        switch (finalGameState)
        {
        case GS_GAMEOVER_LOST:
            // still much better than being stuck after an invalid move
            score *= 2;
            break;
        case GS_GAMEOVER_WON:
            score *= 5;
            break;
        default:
            break;
        }

        return score;
    }

    GameState TicTacToeTrainer::playOneTurn()
    {
        std::vector<double> inputValues;
        m_gameLogic->getNodeNetworkInputValues(inputValues);
        m_nodeNetwork->assignInputValues(inputValues);

        m_nodeNetwork->computeValues();

        std::vector<double> outputValues;
        int bestResult = m_nodeNetwork->getOutputValues(outputValues);

        std::cout << "Output: ";
        for (auto val : outputValues)
        {
            std::cout << val << "  ";
        }

        std::cout << "--> best index: " << bestResult << std::endl;

        if (!m_gameLogic->isValidMove(0, bestResult))
        {
            std::cerr << "Invalid move" << std::endl;
            return GS_GAMEOVER_LOST;
        }

        m_gameLogic->applyMove(0, bestResult);

        const GameState state = m_gameLogic->evaluateBoard();
        std::cout << "Outcome: " << GameLogic::getGameStateDescription(state).c_str() << std::endl;

        return state;
    }
}
