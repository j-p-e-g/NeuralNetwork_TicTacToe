#include "stdafx.h"

#include <assert.h> 
#include <iostream>

#include "TicTacToeTrainer.h"

#include "FileIO/FileManager.h"
#include "Game/GameLogic.h"

namespace Training
{
    using namespace FileIO;
    using namespace Game;
    using namespace NeuralNetwork;
    
    TicTacToeTrainer::TicTacToeTrainer(std::shared_ptr<GameLogic>& gameLogic)
        : BaseTrainer()
        , m_gameLogic(gameLogic)
    {
    }

    TicTacToeTrainer::~TicTacToeTrainer()
    {
    }

    bool TicTacToeTrainer::setupTrainingData()
    {
        TicTacToeLogic::collectInconclusiveFinalGameBoardStates(m_gameStateCollection);
        return true;
    }

    bool TicTacToeTrainer::setupTrainingMethod()
    {
        if (m_useBackpropagation)
        {
            m_trainingMethodHandler = std::make_shared<BackpropagationHandler>(m_nodeNetwork, m_paramManager, m_gameLogic);
        }
        else
        {
            m_trainingMethodHandler = std::make_shared<ParameterEvolutionHandler>(m_nodeNetwork, m_paramManager, m_gameLogic);
        }
        return true;
    }

    NetworkSizeData TicTacToeTrainer::getNetworkSizeData() const
    {
        NetworkSizeData sizeData;
        m_gameLogic->getRequiredNetworkSize(sizeData);
        sizeData.numHiddenNodes = m_numHiddenNodes;

        return sizeData;
    }

    void TicTacToeTrainer::run()
    {
        BaseTrainer::run();

        dumpTrainingStats();
        dumpBestSetImprovementStats();
    }

    void TicTacToeTrainer::handleNetworkComputation(int id, bool isLastIteration)
    {
        m_trainingMethodHandler->iterationStart(id);

        // for each possible permutation on inconclusive last-turn states,
        // try whether the ai makes a valid move
        for (const auto& gameCells : m_gameStateCollection)
        {
            m_gameLogic->setGameCells(gameCells);

            std::shared_ptr<BasePlayer> aiPlayer = std::make_shared<AiPlayer>(id, CellState::CS_PLAYER1, m_nodeNetwork);
            std::vector<double> outputValues;

            const int nextMove = aiPlayer->decideMove(gameCells, outputValues);

            GameState finalState = GameState::GS_GAMEOVER_TIMEOUT;
            if (!m_gameLogic->isValidMove(0, nextMove))
            {
                finalState = GameState::GS_INVALID;
            }

            const double score = computeMatchScore(*aiPlayer, 4, finalState);
            m_trainingMethodHandler->handleTrainingIteration(aiPlayer);
            addScore(*aiPlayer, score, finalState);

        }

        m_trainingMethodHandler->iterationEnd(isLastIteration);
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
        } while (lastStatePlayerA == GS_ONGOING && lastStatePlayerB == GS_ONGOING);

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

        std::vector<double> outputValues;
        const int nextMove = player.decideMove(gameCells, outputValues);

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

        ParamSet pset;
        m_paramManager->getParamSetForId(id, pset);
        buffer << std::endl << "final score: " << pset.score;

        if (pset.error >= 0)
        {
            buffer << std::endl << "avg. error: " << pset.error;
        }

        PRINT_LOG(buffer);
    }

    double TicTacToeTrainer::computeFinalScore(int id)
    {
        auto& found = m_scoreMap.find(id);
        if (found == m_scoreMap.end())
        {
            return 0;
        }

        found->second.finalScore = getOutcomeRatioScoreForId(id) + getAverageScoreForId(id);
        return found->second.finalScore;
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

    bool TicTacToeTrainer::getScoreSetForId(int id, ScoreSet& scoreSet) const
    {
        const auto& found = m_scoreMap.find(id);
        if (found == m_scoreMap.end())
        {
            return false;
        }

        scoreSet = found->second;
        return true;
    }

    void TicTacToeTrainer::dumpTrainingStats() const
    {
        const std::string allSetsFileName = "iteration_scores.csv";
        const std::string bestSetsFileName = "best_iteration_stats.csv";

        std::string relativePath;
        if (!FileManager::getRelativeDataFilePath(allSetsFileName, relativePath))
        {
            return;
        }

        std::ofstream ofs;
        if (!FileManager::openOutFileStream(relativePath, ofs))
        {

            std::ostringstream buffer;
            buffer << "Failed to open file '" << relativePath.c_str() << "' for writing";
            PRINT_ERROR(buffer);
            return;
        }

        ofs << "Iteration, Score, Error" << std::endl;

        for (const auto& iter : m_idsPerIteration)
        {
            for (const auto& id : iter.second)
            {
                ScoreSet score;
                if (getScoreSetForId(id, score))
                {
                    ParamSet pset;
                    m_paramManager->getParamSetForId(id, pset);

                    ofs << (iter.first+1) << ", " << score.finalScore << ", " << pset.error << std::endl;
                }
            }
        }

        std::ostringstream buffer;
        buffer << "dumped training stats to '" << relativePath.c_str() << "'";
        PRINT_LOG(buffer);
    }

    void TicTacToeTrainer::dumpBestSetImprovementStats() const
    {
        const std::string bestSetsFileName = "best_iteration_stats.csv";

        std::string relativePath;
        if (!FileManager::getRelativeDataFilePath(bestSetsFileName, relativePath))
        {
            return;
        }

        std::ofstream ofs;
        if (!FileManager::openOutFileStream(relativePath, ofs))
        {
            std::ostringstream buffer;
            buffer << "Failed to open file '" << relativePath.c_str() << "' for writing";
            PRINT_ERROR(buffer);
            return;
        }

        ofs << "Iteration, Id, Error, Score, OutcomeScore, AvgScore, CountInvalid, CountLost, CountTied, CountWon" << std::endl;

        for (const auto& iter : m_idsPerIteration)
        {
            if (iter.second.empty())
            {
                continue;
            }

            ScoreSet score;
            const int bestId = iter.second[0];
            if (getScoreSetForId(bestId, score))
            {
                ParamSet pset;
                m_paramManager->getParamSetForId(bestId, pset);

                ofs << (iter.first+1) << ", " << bestId 
                    << ", " << pset.error << ", " << score.finalScore 
                    << ", " << getOutcomeRatioScoreForId(bestId) << ", " << getAverageScoreForId(bestId)
                    << ", " << score.invalidCount << ", " << score.lostCount << ", " << score.tiedCount << ", " << score.wonCount << std::endl;
            }
        }

        std::ostringstream buffer;
        buffer << "dumped improvement stats to '" << relativePath.c_str() << "'";
        PRINT_LOG(buffer);
    }
}
