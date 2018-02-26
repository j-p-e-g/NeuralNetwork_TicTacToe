#include "stdafx.h"

#include <assert.h> 
#include <iostream>

#include "TicTacToeTrainer.h"

#include "FileIO/FileManager.h"
#include "GameLogic.h"
#include "NeuralNetwork/NodeNetwork.h"
#include "NeuralNetwork/ParameterManager.h"

namespace Game
{
    using namespace FileIO;
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

    void TicTacToeTrainer::describeTrainer() const
    {
        std::ostringstream buffer;
        buffer << "TicTacToeTrainer: ";
        buffer << std::endl << "  #paramSets: " << m_numParamSets;
        buffer << std::endl << "  #matches: " << m_numMatches;
        buffer << std::endl << "  random param values picked within [" << m_minParamValue << ", " << m_maxParamValue << "]";
        buffer << std::endl;
        PRINT_LOG(buffer);
    }

    bool TicTacToeTrainer::setup()
    {
        describeTrainer();

        m_gameLogic = std::make_shared<TicTacToeLogic>();

        NetworkSizeData sizeData;
        m_gameLogic->getRequiredNetworkSize(sizeData);

        m_nodeNetwork = std::make_shared<NodeNetwork>();
        if (!m_nodeNetwork->createNetwork(sizeData))
        {
            PRINT_ERROR("Network creation failed!");
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

    void TicTacToeTrainer::addScore(const BasePlayer& player, double score)
    {
        auto& found = m_scoreMap.find(player.getId());
        if (found == m_scoreMap.end())
        {
            m_scoreMap.emplace(player.getId(), std::vector<double>({ score }));
        }
        else
        {
            found->second.push_back(score);
        }
    }

    double TicTacToeTrainer::getAverageScoreForId(int id) const
    {
        double score = 0.0;

        const auto& found = m_scoreMap.find(id);
        if (found != m_scoreMap.end())
        {
            assert(!found->second.empty());
            for (const auto& val : found->second)
            {
                score += val;
            }

            score /= found->second.size();
        }

        return score;
    }

    void TicTacToeTrainer::run()
    {
        FileManager::clearLogFile();
        if (!setup())
        {
            PRINT_ERROR("Failed to setup TicTacToeTrainer!");
            return;
        }

        RandomPlayer randomPlayer(-1);

        std::ostringstream buffer;
        for (int k = 0; k < m_numParamSets; k++)
        {
            // reset network parameters
            buffer.clear();
            buffer.str("");
            buffer << "-----------------------------------------------"
                   << std::endl << "Trying parameter set " << k << ": ";
            PRINT_LOG(buffer);

            ParamSet pset;
            m_paramManager->getParamSetForId(k, pset);
            m_nodeNetwork->assignParameters(pset.params);

            AiPlayer aiPlayer(k, m_nodeNetwork);

            // try playing both as the first and the second player
            playMatch(randomPlayer, aiPlayer);
            playMatch(aiPlayer, randomPlayer);

            // also play against yourself
            playMatch(aiPlayer, aiPlayer);
        }

        // update all scores
        for (int k = 0; k < m_numParamSets; k++)
        {
            double avgScore = getAverageScoreForId(k);
            m_paramManager->setScore(k, avgScore);
        }

        m_paramManager->dumpDataToFile();

        std::vector<int> bestSetIds;
        m_paramManager->getParameterSetIdsSortedByScore(bestSetIds);
        assert(!bestSetIds.empty());

        ParamSet pset;
        m_paramManager->getParamSetForId(bestSetIds[0], pset);

        buffer.clear();
        buffer.str("");
        buffer << std::endl << "Best parameter set: " << bestSetIds[0] << " (avg. score: " << pset.score << ")";
        PRINT_LOG(buffer);
    }

    void TicTacToeTrainer::playMatch(BasePlayer& playerA, BasePlayer& playerB)
    {
        std::ostringstream buffer;
        buffer << "New match " << playerA.getPlayerType().c_str() << " vs. " << playerB.getPlayerType().c_str();
        PRINT_LOG(buffer);

        // reset board
        m_gameLogic->initBoard();

        int turnCount = 1;
        GameState lastStatePlayerA = GS_ONGOING;
        GameState lastStatePlayerB = GS_ONGOING;

        bool firstPlayerTurn = true;
        do
        {
            buffer.clear();
            buffer.str("");
            buffer << "Turn " << turnCount << ": " << (firstPlayerTurn ? "player A" : "player B");
            PRINT_LOG(buffer);

            const GameState state = playOneTurn(firstPlayerTurn ? playerA : playerB, firstPlayerTurn);
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

            buffer.clear();
            buffer.str("");
            buffer << "Score player A: " << scorePlayerA;
            PRINT_LOG(buffer);

            addScore(playerA, scorePlayerA);
        }

        if (lastStatePlayerB != GS_ONGOING)
        {
            const int turnCountPlayerB = static_cast<int>(std::floor((float)turnCount / 2));
            double scorePlayerB = computeMatchScore(playerB, turnCountPlayerB, lastStatePlayerB);

            buffer.clear();
            buffer.str("");
            buffer << "Score player B: " << scorePlayerB;
            PRINT_LOG(buffer);

            addScore(playerB, scorePlayerB);
        }
    }

    GameState TicTacToeTrainer::playOneTurn(BasePlayer& player, bool firstPlayer)
    {
        std::vector<CellState> gameCells;
        m_gameLogic->getGameCells(gameCells);
        const int nextMove = player.decideMove(gameCells);

        std::ostringstream buffer;
        buffer << "next move: " << nextMove;
        PRINT_LOG(buffer);

        if (!m_gameLogic->isValidMove(0, nextMove))
        {
            PRINT_LOG("Invalid move");
            return GS_INVALID;
        }

        m_gameLogic->applyMove(firstPlayer ? 0 : 1, nextMove);

        const GameState state = m_gameLogic->evaluateBoard();

        buffer.clear();
        buffer.str("");
        buffer << "Outcome: " << GameLogic::getGameStateDescription(state).c_str();
        PRINT_LOG(buffer);

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
            score = numTurns - 6;
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
