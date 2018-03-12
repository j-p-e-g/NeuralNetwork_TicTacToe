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
        BasePlayer(int id, CellState player);

    public:
        int getId() const { return m_id; }
        CellState getPlayerId() const { return m_player; }
        virtual std::string getPlayerType() const = 0;
        virtual int decideMove(const std::vector<CellState>& gameCells);
        virtual int decideMove(const std::vector<CellState>& gameCells, std::vector<double>& outputValues) = 0;

    protected:
        CellState m_player;

    private:
        int m_id;
    };

    class RandomPlayer
        : public BasePlayer
    {
    public:
        RandomPlayer(int id, CellState player);

    public:
        std::string getPlayerType() const override { return "RandomPlayer"; }

        /// pick a random non-occupied cell
        int decideMove(const std::vector<CellState>& gameCells) override;
        int decideMove(const std::vector<CellState>& gameCells, std::vector<double>& outputValues) override;

    private:
        std::mt19937 m_mt;
    };

    class SemiRandomPlayer
        : public RandomPlayer
    {
    public:
        SemiRandomPlayer(int id, CellState player);

    public:
        std::string getPlayerType() const override { return "SemiRandomPlayer"; }

        /// pick a random non-occupied cell
        int decideMove(const std::vector<CellState>& gameCells) override;
        int decideMove(const std::vector<CellState>& gameCells, std::vector<double>& outputValues) override;

    private:
        std::mt19937 m_mt;
    };

    class AiPlayer
        : public BasePlayer
    {
    public:
        AiPlayer() = delete;
        AiPlayer(int id, CellState player, std::shared_ptr<NeuralNetwork::NodeNetwork>& network);

    public:
        std::string getPlayerType() const override { return "AiPlayer"; }
        int decideMove(const std::vector<CellState>& gameCells) override;
        int decideMove(const std::vector<CellState>& gameCells, std::vector<double>& outputValues) override;
        void getNodeNetworkInputValues(const std::vector<CellState>& gameCells, std::vector<double>& inputValues) const;

    private:
        std::shared_ptr<NeuralNetwork::NodeNetwork>& m_nodeNetwork;
    };
}
