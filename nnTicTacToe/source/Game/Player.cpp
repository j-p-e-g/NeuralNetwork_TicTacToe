#include "stdafx.h"

#include <assert.h>
#include <iostream>

#include "FileIO/FileManager.h"
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

    RandomPlayer::RandomPlayer(int id, CellState player)
        : BasePlayer(id, player)
    {
        std::random_device rd;
        m_mt = std::mt19937(rd());
    }

    int RandomPlayer::decideMove(const std::vector<CellState>& gameCells)
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

    SemiRandomPlayer::SemiRandomPlayer(int id, CellState player)
        : RandomPlayer(id, player)
    {
    }

    int SemiRandomPlayer::decideMove(const std::vector<CellState>& gameCells)
    {
        assert(!gameCells.empty());

        std::vector<int> candidates;

        // first try to find a winning triple
        getTripleCandidates(gameCells, m_player, candidates);

        if (candidates.empty())
        {
            // flip player to try preventing a losing triple
            getTripleCandidates(gameCells, m_player == CS_PLAYER1 ? CS_PLAYER2 : CS_PLAYER1, candidates);

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

    void SemiRandomPlayer::getTripleCandidates(const std::vector<CellState>& gameCells, CellState targeState, std::vector<int>& candidates) const
    {
        // check if we can complete a triple
        for (int k = 0; k < 3; k++)
        {
            // compare rows
            int candidate = getTripleCandidate(gameCells, targeState, 3 * k, 3 * k + 1, 3 * k + 2);
            if (candidate != -1)
            {
                candidates.push_back(candidate);
            }

            // compare columns
            candidate = getTripleCandidate(gameCells, targeState, k, 3 + k, 6 + k);
            if (candidate != -1)
            {
                candidates.push_back(candidate);
            }
        }

        // compare diagonals
        int candidate = getTripleCandidate(gameCells, targeState, 0, 4, 8);
        if (candidate != -1)
        {
            candidates.push_back(candidate);
        }

        candidate = getTripleCandidate(gameCells, targeState, 2, 4, 6);
        if (candidate != -1)
        {
            candidates.push_back(candidate);
        }
    }

    int SemiRandomPlayer::getTripleCandidate(const std::vector<CellState>& gameCells, CellState targetState, int idx1, int idx2, int idx3) const
    {
        assert(idx1 >= 0 && idx1 < gameCells.size());
        assert(idx2 >= 0 && idx2 < gameCells.size());
        assert(idx3 >= 0 && idx3 < gameCells.size());

        const CellState cs1 = gameCells[idx1];
        const CellState cs2 = gameCells[idx2];
        const CellState cs3 = gameCells[idx3];

        if (cs1 == CS_EMPTY && cs2 == targetState && cs2 == cs3)
        {
            return idx1;
        }

        if (cs2 == CS_EMPTY && cs1 == targetState && cs1 == cs3)
        {
            return idx2;
        }

        if (cs3 == CS_EMPTY && cs1 == targetState && cs1 == cs2)
        {
            return idx3;
        }

        return -1;
    }

    AiPlayer::AiPlayer(int id, CellState player, std::shared_ptr<NodeNetwork>& network)
        : BasePlayer(id, player)
        , m_nodeNetwork(network)
    {
    }

    int AiPlayer::decideMove(const std::vector<CellState>& gameCells)
    {
        std::vector<double> inputValues;
        getNodeNetworkInputValues(gameCells, inputValues);

        m_nodeNetwork->assignInputValues(inputValues);
        m_nodeNetwork->computeValues();

        std::vector<double> outputValues;
        int bestResult = m_nodeNetwork->getOutputValues(outputValues);

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
