#include "stdafx.h"

#include <assert.h> 
#include <chrono>
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
        , m_numParamSets(20)
        , m_numIterations(100)
        , m_numMatches(100)
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
        sizeData.numHiddenNodes = std::vector<int>({ 9, 9 });

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

    void TicTacToeTrainer::addScore(const BasePlayer& player, double score, GameState playerGameState)
    {
        auto& found = m_scoreMap.find(player.getId());
        if (found == m_scoreMap.end())
        {
            ScoreSet scoreSet;
            m_scoreMap.emplace(player.getId(), scoreSet);
            found = m_scoreMap.find(player.getId());
        }

        if (found != m_scoreMap.end())
        {
            switch (playerGameState)
            {
            case GS_GAMEOVER_WON:
                found->second.wonCount++;
                break;
            case GS_GAMEOVER_LOST:
                found->second.lostCount++;
                break;
            case GS_GAMEOVER_TIMEOUT:
                found->second.tiedCount++;
                break;
            case GS_INVALID:
                found->second.invalidCount++;
                break;
            default:
                return;
            }

            found->second.scores.push_back(score);
        }
    }

    void TicTacToeTrainer::describeScoreForId(int id) const
    {
        auto& found = m_scoreMap.find(id);
        if (found == m_scoreMap.end())
        {
            return;
        }

        std::ostringstream buffer;
        if (found->second.invalidCount)
        {
            buffer << "#invalid: " << found->second.invalidCount << std::endl;
        }
        if (found->second.wonCount)
        {
            buffer << "#won: " << found->second.wonCount << std::endl;
        }
        if (found->second.lostCount)
        {
            buffer << "#lost: " << found->second.lostCount << std::endl;
        }
        if (found->second.tiedCount)
        {
            buffer << "#tied: " << found->second.tiedCount << std::endl;
        }

        buffer << "Scores: ";
        for (auto score : found->second.scores)
        {
            buffer << score << ",  ";
        }

        buffer << std::endl << "outcome score: " << getOutcomeRatioScoreForId(id);
        buffer << std::endl << "avg. score: " << getAverageScoreForId(id);
        PRINT_LOG(buffer);
    }

    double TicTacToeTrainer::getAverageScoreForId(int id) const
    {
        double score = 0.0;

        const auto& found = m_scoreMap.find(id);
        if (found != m_scoreMap.end())
        {
            assert(!found->second.scores.empty());
            for (const auto& val : found->second.scores)
            {
                score += val;
            }

            score /= found->second.scores.size();
        }

        return score;
    }

    double TicTacToeTrainer::getOutcomeRatioScoreForId(int id) const
    {
        double score = 0.0;

        const auto& found = m_scoreMap.find(id);
        if (found != m_scoreMap.end())
        {
            const int totalCount = found->second.invalidCount + found->second.lostCount + found->second.tiedCount + found->second.wonCount;
            assert(totalCount > 0);
            assert(found->second.scores.size() == totalCount);

            const double quotaValid = 1 - (double)found->second.invalidCount / totalCount;
            const double quotaWon = (double)found->second.wonCount / totalCount;
            const double quotaTied = (double)found->second.tiedCount / totalCount;
            score = 100 * quotaValid + 10 * quotaWon + quotaTied;
        }

        return score;
    }

    void TicTacToeTrainer::run()
    {
        const auto processStart = std::chrono::high_resolution_clock::now();

        FileManager::clearLogFile();
        if (!setup())
        {
            PRINT_ERROR("Failed to setup TicTacToeTrainer!");
            return;
        }

        for (int i = 0; i < m_numIterations; i++)
        {
            std::cout << "training iteration " << i << std::endl;
            const bool requiresFurtherEvolution = (i < m_numIterations - 1);
            handleTrainingIteration(requiresFurtherEvolution);
        }

        const auto processEnd = std::chrono::high_resolution_clock::now();
        const std::chrono::duration<double> elapsedSeconds = processEnd - processStart;

        std::ostringstream buffer;
        buffer << "Time taken: " << elapsedSeconds.count() << " seconds";
        std::cout << buffer.str();
        PRINT_LOG(buffer);

        m_paramManager->dumpDataToFile();
    }

    void TicTacToeTrainer::handleTrainingIteration(bool requiresFurtherEvolution)
    {
        std::ostringstream buffer;

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

            {
                // play N matches with the AI as the first player
                AiPlayer aiPlayer(id, CellState::CS_PLAYER1, m_nodeNetwork);
                SemiRandomPlayer randomPlayer(-1, CellState::CS_PLAYER2);

                for (int k = 0; k < m_numMatches; k++)
                {
                    playMatch(aiPlayer, randomPlayer);
                }
            }

            {
                // play N matches with the AI as the second player
                SemiRandomPlayer randomPlayer(-1, CellState::CS_PLAYER1);
                AiPlayer aiPlayer(id, CellState::CS_PLAYER2, m_nodeNetwork);

                for (int k = 0; k < m_numMatches; k++)
                {
                    playMatch(randomPlayer, aiPlayer);
                }
            }

            describeScoreForId(id);

            // update score
            const double newScore = getOutcomeRatioScoreForId(id) + getAverageScoreForId(id);
            m_paramManager->setScore(id, newScore);
        }

        m_paramManager->dumpDataToFile();

        std::vector<int> bestSetIds;
        m_paramManager->getParameterSetIdsSortedByScore(bestSetIds);
        assert(!bestSetIds.empty());

        ParamSet pset;
        m_paramManager->getParamSetForId(bestSetIds[0], pset);

        buffer.clear();
        buffer.str("");
        buffer << std::endl << "Best parameter set: " << bestSetIds[0] << ", with score: " << pset.score;
        PRINT_LOG(buffer);

        if (requiresFurtherEvolution)
        {
            handleParamSetEvolution();
        }
    }

    void TicTacToeTrainer::playMatch(BasePlayer& playerA, BasePlayer& playerB)
    {
        //std::ostringstream buffer;
        //buffer << "New match " << playerA.getPlayerType().c_str() << " vs. " << playerB.getPlayerType().c_str();
        //PRINT_LOG(buffer);

        // reset board
        m_gameLogic->initBoard();

        int turnCount = 1;
        GameState lastStatePlayerA = GS_ONGOING;
        GameState lastStatePlayerB = GS_ONGOING;

        bool firstPlayerTurn = true;
        do
        {
            //buffer.clear();
            //buffer.str("");
            //buffer << "Turn " << turnCount << ": " << (firstPlayerTurn ? "player A" : "player B");
            //PRINT_LOG(buffer);

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

                // flip win/loss for other player
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

                // both players are tied
                if (lastStatePlayerA == GS_GAMEOVER_TIMEOUT || lastStatePlayerB == GS_GAMEOVER_TIMEOUT)
                {
                    lastStatePlayerA = GS_GAMEOVER_TIMEOUT;
                    lastStatePlayerB = GS_GAMEOVER_TIMEOUT;
                }
            }
        }
        while (lastStatePlayerA == GS_ONGOING && lastStatePlayerB == GS_ONGOING);

        if (lastStatePlayerA != GS_ONGOING)
        {
            const int turnCountPlayerA = static_cast<int>(std::ceil((float)turnCount / 2));
            double scorePlayerA = computeMatchScore(playerA, turnCountPlayerA, lastStatePlayerA);

            //buffer.clear();
            //buffer.str("");
            //buffer << "Score player A: " << scorePlayerA;
            //PRINT_LOG(buffer);

            addScore(playerA, scorePlayerA, lastStatePlayerA);
        }

        if (lastStatePlayerB != GS_ONGOING)
        {
            const int turnCountPlayerB = static_cast<int>(std::floor((float)turnCount / 2));
            double scorePlayerB = computeMatchScore(playerB, turnCountPlayerB, lastStatePlayerB);

            //buffer.clear();
            //buffer.str("");
            //buffer << "Score player B: " << scorePlayerB;
            //PRINT_LOG(buffer);

            addScore(playerB, scorePlayerB, lastStatePlayerB);
        }
    }

    GameState TicTacToeTrainer::playOneTurn(BasePlayer& player, bool firstPlayer)
    {
        std::vector<CellState> gameCells;
        m_gameLogic->getGameCells(gameCells);
        const int nextMove = player.decideMove(gameCells);

        //std::ostringstream buffer;
        //buffer << "next move: " << nextMove;
        //PRINT_LOG(buffer);

        if (!m_gameLogic->isValidMove(0, nextMove))
        {
            //PRINT_LOG("Invalid move");
            return GS_INVALID;
        }

        m_gameLogic->applyMove(firstPlayer ? 0 : 1, nextMove);

        const GameState state = m_gameLogic->evaluateBoard();

        //buffer.clear();
        //buffer.str("");
        //buffer << "Outcome: " << GameLogic::getGameStateDescription(state).c_str();
        //PRINT_LOG(buffer);

        return state;
    }

    double TicTacToeTrainer::computeMatchScore(BasePlayer& player, int numTurns, GameState finalGameState)
    {
        switch (finalGameState)
        {
        case GS_INVALID:
            // made an invalid move
            // the penalty is smaller the later this happens
            // score between -10 and -2
            return 2 * (numTurns - 6);
        case GS_ONGOING:
            // the other player made an invalid move
            break;
        case GS_GAMEOVER_LOST:
            // still much better than being stuck after an invalid move
            // the bonus is larger the later this happens
            // score between 1 and 5
            return numTurns;
        case GS_GAMEOVER_TIMEOUT:
            // the game ended in a tie
            return 10;
        case GS_GAMEOVER_WON:
            // the bonus is larger the earlier this happens
            // score between 11 and 15
            return (6 - numTurns) + 10;
        }

        return 0;
    }

    void TicTacToeTrainer::handleParamSetEvolution()
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
                //m_paramManager->removeParameterSetForId(id);
            }
        }
    }

}
