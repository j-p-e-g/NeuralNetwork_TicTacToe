#include "stdafx.h"

#include <assert.h> 
#include <chrono>
#include <iostream>

#include "3rdparty/json/json.hpp"

#include "TicTacToeTrainer.h"

#include "FileIO/FileManager.h"
#include "GameLogic.h"
#include "NeuralNetwork/NodeNetwork.h"
#include "NeuralNetwork/ParameterManager.h"

#define JUST_TEST_VALIDITY 1

namespace Game
{
    using namespace FileIO;
    using namespace NeuralNetwork;
    using json = nlohmann::json;

    const std::string CONFIG_FILE_NAME = "configs.json";

    TicTacToeTrainer::~TicTacToeTrainer()
    {
        if (m_initialized)
        {
            m_nodeNetwork->destroyNetwork();
        }
    }

    bool TicTacToeTrainer::readConfigValues()
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

    void TicTacToeTrainer::describeTrainer() const
    {
        std::ostringstream buffer;
        buffer << "TicTacToeTrainer: ";
        buffer << std::endl << "  #paramSets: " << m_numParamSets;
        buffer << std::endl << "  #matches: " << m_numMatches;
        buffer << std::endl << "  #iterations: " << m_numIterations;
        buffer << std::endl;
        PRINT_LOG(buffer);
    }

    bool TicTacToeTrainer::setup()
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

        m_gameLogic = std::make_shared<TicTacToeLogic>();

        // setup network
        NetworkSizeData sizeData;
        m_gameLogic->getRequiredNetworkSize(sizeData);
        sizeData.numHiddenNodes = m_numHiddenNodes;

        m_nodeNetwork = std::make_shared<NodeNetwork>();
        if (!m_nodeNetwork->createNetwork(sizeData, m_activationFunctionType))
        {
            PRINT_ERROR("Network creation failed!");
            return false;
        }

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

    void TicTacToeTrainer::run()
    {
        const auto processStart = std::chrono::high_resolution_clock::now();

        FileManager::clearLogFile();
        if (!setup())
        {
            PRINT_ERROR("Failed to setup TicTacToeTrainer!");
            return;
        }

#ifdef JUST_TEST_VALIDITY
        TicTacToeLogic::collectInconclusiveFinalGameBoardStates(m_gameStateCollection);
#endif

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

        dumpTrainingStats();
        dumpBestSetImprovementStats();
    }

    void TicTacToeTrainer::handleTrainingIteration(int iteration)
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

#ifdef JUST_TEST_VALIDITY
            {
                // for each possible permutation on inconclusive last-turn states,
                // try whether the ai makes a valid move
                for (const auto& gameCells : m_gameStateCollection)
                {
                    m_gameLogic->setGameCells(gameCells);

                    AiPlayer aiPlayer(id, CellState::CS_PLAYER1, m_nodeNetwork);
                    const int nextMove = aiPlayer.decideMove(gameCells);

                    GameState finalState = GameState::GS_GAMEOVER_TIMEOUT;
                    if (!m_gameLogic->isValidMove(0, nextMove))
                    {
                        finalState = GameState::GS_INVALID;
                    }

                    const double score = computeMatchScore(aiPlayer, 4, finalState);
                    addScore(aiPlayer, score, finalState);
                }
            }
#else
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
#endif

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

        ofs << "Iteration, Score" << std::endl;

        for (const auto& iter : m_idsPerIteration)
        {
            for (const auto& id : iter.second)
            {
                ScoreSet score;
                if (getScoreSetForId(id, score))
                {
                    ofs << (iter.first+1) << ", " << score.finalScore << std::endl;
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

        ofs << "Iteration, Id, Score, OutcomeScore, AvgScore, CountInvalid, CountLost, CountTied, CountWon" << std::endl;

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
                ofs << (iter.first+1) << ", " << bestId << ", " << score.finalScore << ", " << getOutcomeRatioScoreForId(bestId) << ", " << getAverageScoreForId(bestId)
                    << ", " << score.invalidCount << ", " << score.lostCount << ", " << score.tiedCount << ", " << score.wonCount << std::endl;
            }
        }

        std::ostringstream buffer;
        buffer << "dumped improvement stats to '" << relativePath.c_str() << "'";
        PRINT_LOG(buffer);
    }

    bool TicTacToeTrainer::handleOptionValidation() const
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

}
