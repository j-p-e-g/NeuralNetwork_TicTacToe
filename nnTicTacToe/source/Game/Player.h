#pragma once

#include <random>
#include <vector>

#include "General/Globals.h"
#include "NeuralNetwork/NodeNetwork.h"

namespace Game
{
    class BasePlayer
    {
    public:
        BasePlayer(int id);

    public:
        int getId() const { return m_id; }
        virtual int decideMove(const std::vector<CellState>& gameCells) = 0;

    private:
        int m_id;
    };

    class RandomPlayer
        : public BasePlayer
    {
    public:
        RandomPlayer(int id);

    public:
        /// pick a random non-occupied cell
        int decideMove(const std::vector<CellState>& gameCells) override;

    private:
        std::mt19937 m_mt;
    };

    class AiPlayer
        : public BasePlayer
    {
    public:
        AiPlayer() = delete;
        AiPlayer(int id, std::shared_ptr<NeuralNetwork::NodeNetwork>& network);

    public:
        int decideMove(const std::vector<CellState>& gameCells) override;
        void getNodeNetworkInputValues(const std::vector<CellState>& gameCells, std::vector<double>& inputValues) const;

    private:
        std::shared_ptr<NeuralNetwork::NodeNetwork>& m_nodeNetwork;
    };
}
