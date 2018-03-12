#include "stdafx.h"

#include <assert.h>
#include <iostream>

#include "FileIO/FileManager.h"
#include "GameLogic.h"
#include "Player.h"

namespace Game
{
    using namespace FileIO;
    using namespace NeuralNetwork;

    BasePlayer::BasePlayer(int id, CellState player)
        : m_id(id)
        , m_player(player)
    {
    }

    int BasePlayer::decideMove(const std::vector<CellState>& gameCells)
    {
        std::vector<double> outputCells;
        return decideMove(gameCells, outputCells);
    }

    // RandomPlayer:
    // randomly picks an empty cell
    RandomPlayer::RandomPlayer(int id, CellState player)
        : BasePlayer(id, player)
    {
        std::random_device rd;
        m_mt = std::mt19937(rd());
    }

    int RandomPlayer::decideMove(const std::vector<CellState>& gameCells)
    {
        return BasePlayer::decideMove(gameCells);
    }

    int RandomPlayer::decideMove(const std::vector<CellState>& gameCells, std::vector<double>& outputValues)
    {
        assert(!gameCells.empty());

        std::vector<int> candidates;
        for (unsigned int k = 0; k < gameCells.size(); k++)
        {
            if (gameCells[k] == CellState::CS_EMPTY)
            {
                candidates.push_back(k);
            }
        }

        assert(!candidates.empty());
        if (candidates.empty())
        {
            return -1;
        }

        std::uniform_int_distribution<int> rndDist(0, static_cast<int>(candidates.size())-1);
        const int randomIndex = rndDist(m_mt);
        return candidates[randomIndex];
    }

    // SemiRandomPlayer:
    // tries to close any open triples, otherwise randomly picks an empty cell
    SemiRandomPlayer::SemiRandomPlayer(int id, CellState player)
        : RandomPlayer(id, player)
    {
    }

    int SemiRandomPlayer::decideMove(const std::vector<CellState>& gameCells)
    {
        return RandomPlayer::decideMove(gameCells);
    }

    int SemiRandomPlayer::decideMove(const std::vector<CellState>& gameCells, std::vector<double>& outputValues)
    {
        assert(!gameCells.empty());

        std::vector<int> candidates;

        // first try to find a winning triple
        TicTacToeLogic::getTripleCandidates(gameCells, m_player, candidates);

        if (candidates.empty())
        {
            // flip player to try preventing a losing triple
            TicTacToeLogic::getTripleCandidates(gameCells, m_player == CS_PLAYER1 ? CS_PLAYER2 : CS_PLAYER1, candidates);

            if (candidates.empty())
            {
                // if no triple is possible, choose randomly
                return RandomPlayer::decideMove(gameCells);
            }
        }

        std::uniform_int_distribution<int> rndDist(0, static_cast<int>(candidates.size()) - 1);
        const int randomIndex = rndDist(m_mt);
        return candidates[randomIndex];
    }

    // AI player:
    // uses neural networks to find a solution
    AiPlayer::AiPlayer(int id, CellState player, std::shared_ptr<NodeNetwork>& network)
        : BasePlayer(id, player)
        , m_nodeNetwork(network)
    {
    }

    int AiPlayer::decideMove(const std::vector<CellState>& gameCells)
    {
        return BasePlayer::decideMove(gameCells);
    }

    int AiPlayer::decideMove(const std::vector<CellState>& gameCells, std::vector<double>& outputValues)
    {
        std::vector<double> inputValues;
        getNodeNetworkInputValues(gameCells, inputValues);

        m_nodeNetwork->assignInputValues(inputValues);
        m_nodeNetwork->computeValues();

        const int bestResult = m_nodeNetwork->getOutputValues(outputValues);

        //std::ostringstream buffer;
        //buffer << "Output: ";
        //for (auto val : outputValues)
        //{
        //    buffer << val << "  ";
        //}

        //buffer << "--> best index: " << bestResult;
        //PRINT_LOG(buffer);

        return bestResult;
    }

    void AiPlayer::getNodeNetworkInputValues(const std::vector<CellState>& gameCells, std::vector<double>& inputValues) const
    {
        inputValues.clear();

        for (const auto& cell : gameCells)
        {
            if (cell == CellState::CS_EMPTY)
            {
                inputValues.push_back(-1.0);
                inputValues.push_back(-1.0);
            }
            else if (cell == m_player)
            {
                // own cells
                inputValues.push_back(1.0);
                inputValues.push_back(-1.0);
            }
            else
            {
                // other player cells
                inputValues.push_back(-1.0);
                inputValues.push_back(1.0);
            }
        }
    }
}
