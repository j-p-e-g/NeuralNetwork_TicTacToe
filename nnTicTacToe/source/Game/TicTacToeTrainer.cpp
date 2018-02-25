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

            AiPlayer playerA(0, m_nodeNetwork);
            RandomPlayer playerB(1);

            playMatch(playerA, playerB);
        }

        std::vector<int> bestSetIds;
        m_paramManager->getParameterSetIdsSortedByScore(bestSetIds);
        assert(!bestSetIds.empty());

        ParamSet pset;
        m_paramManager->getParamSetForId(bestSetIds[0], pset);

        std::cout << std::endl << "Best parameter set: " << bestSetIds[0] << " (score: " << pset.score << ")" << std::endl;
    }

    void TicTacToeTrainer::playMatch(BasePlayer& playerA, BasePlayer& playerB)
    {
        // reset board
        m_gameLogic->initBoard();

        int turnCount = 1;
        GameState lastStatePlayerA = GS_ONGOING;
        GameState lastStatePlayerB = GS_ONGOING;

        bool firstPlayerTurn = true;

        do
        {
            std::cout << "Turn " << turnCount << ": " << (firstPlayerTurn ? "player A" : "player B") << std::endl;

            const GameState state = playOneTurn(firstPlayerTurn ? playerA : playerB);
            turnCount++;

            if (state == GS_ONGOING)
            {
                firstPlayerTurn = !firstPlayerTurn;
            }
            else
            {
                if (firstPlayerTurn)
                {
                    lastStatePlayerA = state;
                }
                else
                {
                    lastStatePlayerB = state;
                }

                if (lastStatePlayerA == GS_GAMEOVER_WON || lastStatePlayerB == GS_GAMEOVER_LOST)
                {
                    lastStatePlayerA = GS_GAMEOVER_WON;
                    lastStatePlayerB = GS_GAMEOVER_LOST;
                }
                else if (lastStatePlayerA == GS_GAMEOVER_LOST || lastStatePlayerB == GS_GAMEOVER_WON)
                {
                    lastStatePlayerA = GS_GAMEOVER_LOST;
                    lastStatePlayerB = GS_GAMEOVER_WON;
                }
            }
        }
        while (lastStatePlayerA == GS_ONGOING && lastStatePlayerB == GS_ONGOING);

        if (lastStatePlayerA != GS_ONGOING)
        {
            const int turnCountPlayerA = static_cast<int>(std::ceil((float)turnCount / 2));
            double scorePlayerA = computeMatchScore(playerA, turnCountPlayerA, lastStatePlayerA);
            std::cout << "Score player A: " << scorePlayerA << std::endl;
            m_paramManager->setScore(playerA.getId(), scorePlayerA);
        }

        if (lastStatePlayerB != GS_ONGOING)
        {
            const int turnCountPlayerB = static_cast<int>(std::floor((float)turnCount / 2));
            double scorePlayerB = computeMatchScore(playerB, turnCountPlayerB, lastStatePlayerB);
            std::cout << "Score player B: " << scorePlayerB << std::endl;
            m_paramManager->setScore(playerB.getId(), scorePlayerB);
        }
    }

    GameState TicTacToeTrainer::playOneTurn(BasePlayer& player)
    {
        std::vector<CellState> gameCells;
        m_gameLogic->getGameCells(gameCells);
        const int nextMove = player.decideMove(gameCells);
        std::cout << "next move: " << nextMove << std::endl;

        if (!m_gameLogic->isValidMove(0, nextMove))
        {
            std::cerr << "Invalid move" << std::endl;
            return GS_INVALID;
        }

        m_gameLogic->applyMove(0, nextMove);

        const GameState state = m_gameLogic->evaluateBoard();
        std::cout << "Outcome: " << GameLogic::getGameStateDescription(state).c_str() << std::endl;

        return state;
    }

    double TicTacToeTrainer::computeMatchScore(BasePlayer& player, int numTurns, GameState finalGameState)
    {
        double score = 0.0;

        switch (finalGameState)
        {
        case GS_INVALID:
            // made an invalid move
            // the penalty is smaller the later this happens
            score -= numTurns;
            break;
        case GS_ONGOING:
            // the other player made an invalid move
            break;
        case GS_GAMEOVER_LOST:
            // still much better than being stuck after an invalid move
            // the bonus is larger the later this happens
            score += numTurns;
            break;
        case GS_GAMEOVER_WON:
            // the bonus is larger the earlier this happens
            score += (6 - numTurns) + 10;
            break;
        }

        return score;
    }
}
