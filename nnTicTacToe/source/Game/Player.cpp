#include "stdafx.h"

#include <assert.h>
#include <iostream>

#include "Player.h"

namespace Game
{
    using namespace NeuralNetwork;

    BasePlayer::BasePlayer(int id)
        : m_id(id)
    {
    }

    RandomPlayer::RandomPlayer(int id)
        : BasePlayer(id)
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

    AiPlayer::AiPlayer(int id, std::shared_ptr<NodeNetwork>& network)
        : BasePlayer(id)
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

        std::cout << "Output: ";
        for (auto val : outputValues)
        {
            std::cout << val << "  ";
        }

        std::cout << "--> best index: " << bestResult << std::endl;

        return bestResult;
    }

    void AiPlayer::getNodeNetworkInputValues(const std::vector<CellState>& gameCells, std::vector<double>& inputValues) const
    {
        inputValues.clear();

        for (const auto& cell : gameCells)
        {
            switch (cell)
            {
            case CellState::CS_EMPTY:
                inputValues.push_back(-1.0);
                inputValues.push_back(-1.0);
                break;
            case CellState::CS_PLAYER1:
                inputValues.push_back(1.0);
                inputValues.push_back(-1.0);
                break;
            case CellState::CS_PLAYER2:
                inputValues.push_back(-1.0);
                inputValues.push_back(1.0);
                break;
            }
        }
    }
}
