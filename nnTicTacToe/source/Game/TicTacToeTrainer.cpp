#include "stdafx.h"

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

        m_initialized = true;
        return true;
    }

    void TicTacToeTrainer::setupTicTacToeTest()
    {
        if (!m_initialized)
        {
            return;
        }

        int roundCount = 1;
        for (int k = 0; k < m_numMatches; k++)
        {
            std::cout << std::endl << "Round " << roundCount << ": " << std::endl;

            playOneMatch();
            roundCount++;
        }
    }

    void TicTacToeTrainer::playOneMatch()
    {
        // reset board
        m_gameLogic->initBoard();

        // reset parameters
        ParamSet pset;
        m_paramManager->fillWithRandomValues(pset.params);
        m_nodeNetwork->assignParameters(pset.params);

        int turnCount = 1;
        GameState state = GS_ONGOING;
        do
        {
            std::cout << "Turn " << turnCount << ": " << std::endl;
            state = playOneTurn();
            turnCount++;
        }
        while (state == GS_ONGOING);
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

    void TicTacToeTrainer::run()
    {
        if (!m_initialized)
        {
            std::cerr << "TicTacToeTrainer was not setup correctly. Call setup() to initialize." << std::endl;
            return;
        }

        setupTicTacToeTest();
    }
}
